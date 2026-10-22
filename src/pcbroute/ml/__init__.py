"""ML-assisted net ordering and congestion prediction for routing."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from ..ir import Design, Net
from ..geometry import GeometryKernel, KernelConfig
from ..routing import GridRouter, RouterConfig


@dataclass
class NetFeatures:
    """Features extracted for a net for ML prediction."""
    net_name: str
    num_pins: int
    bbox_width_mm: float
    bbox_height_mm: float
    hpwl_mm: float
    pin_density: float
    num_layers: int
    net_kind: int  # 0=signal, 1=power, 2=ground, 3=clock
    width_mm: float
    clearance_mm: float
    has_existing_copper: bool
    estimated_via_count: int
    escape_directions: int
    
    # Congestion prediction features
    competing_nets: int = 0
    local_obstacle_density: float = 0.0
    free_cell_ratio: float = 0.0
    estimated_congestion: float = 0.0


@dataclass
class NetRoutingLabel:
    """Ground truth label for net routing difficulty."""
    net_name: str
    route_success: bool
    path_length_mm: float
    via_count: int
    steps_taken: int
    status: str  # "routed", "partial", "failed"
    
    # Congestion target
    congestion_score: float = 0.0  # Normalized congestion metric
    max_congestion: float = 0.0
    congestion_cells: int = 0


class NetOrderingMLP(nn.Module):
    """Small MLP for predicting net routing difficulty."""
    
    def __init__(self, input_dim: int = 13, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class CongestionPredictionMLP(nn.Module):
    """Small MLP for predicting routing congestion."""
    
    def __init__(self, input_dim: int = 17, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def extract_net_features(design: Design, net: Net, kernel: GeometryKernel) -> NetFeatures:
    """Extract features for a net."""
    pins = [p for p in design.pins() if p.net == net.name]
    num_pins = len(pins)
    
    if num_pins == 0:
        return NetFeatures(
            net_name=net.name, num_pins=0, bbox_width_mm=0, bbox_height_mm=0,
            hpwl_mm=0, pin_density=0, num_layers=0, net_kind=0,
            width_mm=0, clearance_mm=0, has_existing_copper=False,
            estimated_via_count=0, escape_directions=0,
            competing_nets=0, local_obstacle_density=0.0,
            free_cell_ratio=0.0, estimated_congestion=0.0
        )
    
    # Bounding box
    xs = [p.position.x for p in pins]
    ys = [p.position.y for p in pins]
    bbox_width_mm = (max(xs) - min(xs)) / 1_000_000.0
    bbox_height_mm = (max(ys) - min(ys)) / 1_000_000.0
    
    # Half-perimeter wirelength
    hpwl_mm = bbox_width_mm + bbox_height_mm
    
    # Pin density (pins per mm^2)
    area_mm2 = max(0.01, bbox_width_mm * bbox_height_mm)
    pin_density = num_pins / area_mm2
    
    # Available layers
    layer_mask = set()
    for p in pins:
        layer_mask.update(p.layers)
    num_layers = len(layer_mask)
    
    # Net kind
    kind_map = {"signal": 0, "power": 1, "ground": 2, "clock": 3}
    net_kind = kind_map.get(net.kind.value if hasattr(net.kind, 'value') else str(net.kind), 0)
    
    # Check existing copper
    has_existing = False
    for seg in design.copper.segments:
        if seg.net == net.name:
            has_existing = True
            break
    
    # Estimate via count (simplified)
    estimated_vias = max(0, num_pins - 2)  # rough estimate
    
    # Escape directions (how many free directions from pins)
    escape_dirs = 0
    for pin in pins:
        for layer in pin.layers:
            if layer in kernel.layer_ids:
                x, y = pin.position.x, pin.position.y
                # Check 4 directions
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = x + dx * 500_000, y + dy * 500_000
                    if kernel.is_trace_legal(layer, (x, y), (nx, ny), net.width_nm, net.name):
                        escape_dirs += 1
    
    # Congestion features
    # Competing nets: number of other routable nets that overlap in bbox
    competing = 0
    for other_net in design.nets:
        if other_net.name == net.name or len(other_net.pins) < 2:
            continue
        other_pins = [p for p in design.pins() if p.net == other_net.name]
        other_xs = [p.position.x for p in other_pins]
        other_ys = [p.position.y for p in other_pins]
        if (max(xs) >= min(other_xs) and min(xs) <= max(other_xs) and
            max(ys) >= min(other_ys) and min(ys) <= max(other_ys)):
            competing += 1
    
    # Local obstacle density near the net's bbox
    kernel_config = KernelConfig()
    temp_kernel = GeometryKernel(design, kernel_config)
    bbox_nm = (min(xs), min(ys), max(xs), max(ys))
    expanded_bbox = (
        bbox_nm[0] - 1_000_000,  # expand by 1mm
        bbox_nm[1] - 1_000_000,
        bbox_nm[2] + 1_000_000,
        bbox_nm[3] + 1_000_000,
    )
    
    obstacle_count = 0
    total_cells = 0
    for layer in kernel.layer_ids:
        items = temp_kernel.items_in_bbox(layer, 
            type('BBox', (), {'x0': expanded_bbox[0], 'y0': expanded_bbox[1], 'x1': expanded_bbox[2], 'y1': expanded_bbox[3]})())
        obstacle_count += len([i for i in items if i.kind in ('track', 'via', 'pad', 'keepout')])
        # Estimate cell count in bbox area
        cell_area = temp_kernel.config.cell_nm * temp_kernel.config.cell_nm
        bbox_area = (expanded_bbox[2] - expanded_bbox[0]) * (expanded_bbox[3] - expanded_bbox[1])
        total_cells += max(1, bbox_area // cell_area)
    
    local_obstacle_density = obstacle_count / max(1, total_cells)
    
    # Free cell ratio
    free_cell_ratio = max(0.0, 1.0 - local_obstacle_density)
    
    # Estimated congestion (combining competing nets and obstacle density)
    estimated_congestion = (competing * 0.5 + local_obstacle_density * 10.0)
    
    return NetFeatures(
        net_name=net.name,
        num_pins=num_pins,
        bbox_width_mm=bbox_width_mm,
        bbox_height_mm=bbox_height_mm,
        hpwl_mm=hpwl_mm,
        pin_density=pin_density,
        num_layers=num_layers,
        net_kind=net_kind,
        width_mm=net.width_nm / 1_000_000.0,
        clearance_mm=net.clearance_nm / 1_000_000.0,
        has_existing_copper=has_existing,
        estimated_via_count=estimated_vias,
        escape_directions=escape_dirs,
        competing_nets=competing,
        local_obstacle_density=local_obstacle_density,
        free_cell_ratio=free_cell_ratio,
        estimated_congestion=estimated_congestion,
    )


def features_to_array(f: NetFeatures) -> np.ndarray:
    """Convert NetFeatures to numpy array."""
    return np.array([
        f.num_pins,
        f.bbox_width_mm,
        f.bbox_height_mm,
        f.hpwl_mm,
        f.pin_density,
        f.num_layers,
        f.net_kind,
        f.width_mm,
        f.clearance_mm,
        float(f.has_existing_copper),
        f.estimated_via_count,
        f.escape_directions,
        f.competing_nets,
        f.local_obstacle_density,
        f.free_cell_ratio,
        f.estimated_congestion,
        1.0,  # bias term
    ], dtype=np.float32)


def generate_synthetic_training_data(
    num_boards: int = 50,
    seed: int = 42,
    output_dir: str = "ml_training_data"
) -> list[tuple[NetFeatures, NetRoutingLabel]]:
    """Generate synthetic training data by routing random boards."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed)
    training_data = []
    
    for board_idx in range(num_boards):
        print(f"  Board {board_idx+1}/{num_boards}...", flush=True)
        # Create a simple synthetic board
        design = _create_synthetic_board(rng, board_idx)
        print(f"    Created design with {len(design.nets)} nets", flush=True)
        
        # Route with deterministic router (fast config)
        config = RouterConfig(grid_nm=1_000_000, max_steps_per_net=5000, random_seed=seed + board_idx)
        kernel = GeometryKernel(design, KernelConfig(allow_blind_buried=config.allow_blind_buried))
        router = GridRouter(design, config)
        
        for net in design.nets:
            if len(net.pins) < 2:
                continue
            
            features = extract_net_features(design, net, kernel)
            result = router.route_net(net)
            
            # Compute congestion metrics from routing result
            congestion_cells = 0
            max_congestion = 0.0
            for seg in result.segments:
                a, b, layer, _ = seg
                gx0 = int((a.x - router.gx0) / config.grid_nm)
                gy0 = int((a.y - router.gy0) / config.grid_nm)
                gx1 = int((b.x - router.gx0) / config.grid_nm)
                gy1 = int((b.y - router.gy0) / config.grid_nm)
                # Approximate congestion by segment length
                length_cells = max(abs(gx1 - gx0), abs(gy1 - gy0)) + 1
                congestion_cells += length_cells
            
            # Estimate max congestion from router's congestion map
            max_congestion = float(router.congestion.max()) if router.congestion.size > 0 else 0.0
            
            # Normalize congestion score (0-1)
            congestion_score = min(1.0, congestion_cells / 100.0 + max_congestion / 10.0)
            
            label = NetRoutingLabel(
                net_name=net.name,
                route_success=result.status.value == "routed",
                path_length_mm=result.path_length_nm / 1_000_000.0,
                via_count=result.via_count,
                steps_taken=result.steps_taken,
                status=result.status.value,
                congestion_score=congestion_score,
                max_congestion=max_congestion,
                congestion_cells=congestion_cells,
            )
            
            features = extract_net_features(design, net, kernel)
            training_data.append((features, label))
    
    # Save to disk
    _save_training_data(training_data, output_dir)
    
    return training_data


def _create_synthetic_board(rng: np.random.Generator, board_idx: int) -> Design:
    """Create a synthetic board for training."""
    from ..ir import (Board, Component, Copper, CopperLayer, Cutout, Design, LayerKind, Net, NetClass, NetKind,
                      PadShape, Pin, PinKind, Polygon, Pt, Rules, Side, Stackup, TrackSeg, Via, ViaKind, ViaType, Zone, ZoneKind)
    from ..ir.geometry import mm_to_nm, Pt as IPPt
    
    def rect_ring(x0, y0, x1, y1):
        return tuple(IPPt(mm_to_nm(x), mm_to_nm(y)) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    
    # Random board size
    board_w = rng.uniform(50, 150)
    board_h = rng.uniform(40, 120)
    
    # 4-layer stackup
    kinds = [LayerKind.SIGNAL, LayerKind.GROUND, LayerKind.POWER, LayerKind.SIGNAL]
    sides = ["top", "inner", "inner", "bottom"]
    stack = Stackup(tuple(CopperLayer(
        f"L{i}", i, kinds[i], 35_000, n, sides[i])
        for i, n in enumerate(("F.Cu", "In1.Cu", "In2.Cu", "B.Cu"))))
    
    board = Board(outline=rect_ring(0, 0, board_w, board_h), stackup=stack)
    
    # Random number of nets (2-6), each with exactly 2 pins
    num_nets = rng.integers(2, 7)
    net_names = [f"NET{i}" for i in range(num_nets)]
    
    # Ensure at least one power and one ground
    if num_nets >= 3:
        net_names[0] = "GND"
        net_names[1] = "VCC"
    
    # Create nets and pins
    pins = []
    nets = []
    components = []
    
    for net_idx, net_name in enumerate(net_names):
        # Always exactly 2 pins per net
        num_pins = 2
        kind = NetKind.GROUND if net_name == "GND" else (NetKind.POWER if net_name == "VCC" else NetKind.SIGNAL)
        
        net_pins = []
        for pin_idx in range(num_pins):
            # Random position
            x = rng.uniform(5, board_w - 5)
            y = rng.uniform(5, board_h - 5)
            
            # Random layers
            if net_name in ("GND", "VCC"):
                layers = ("L0", "L1", "L2", "L3")
                pin_kind = PinKind.THROUGH
            else:
                layers = ("L0",)
                pin_kind = PinKind.SMD
            
            pin_id = f"{net_name}.{pin_idx}"
            pin = Pin(
                id=pin_id, ref=f"U{net_idx}", number=str(pin_idx), function="", net=net_name,
                layers=layers, position=Pt(mm_to_nm(x), mm_to_nm(y)),
                shape=PadShape.CIRCLE if pin_kind == PinKind.THROUGH else PadShape.RECT,
                width_nm=mm_to_nm(1.0), height_nm=mm_to_nm(1.0), rotation_deg=0.0,
                kind=pin_kind, drill_nm=mm_to_nm(0.5) if pin_kind == PinKind.THROUGH else None,
                role=kind, source_id=""
            )
            pins.append(pin)
            net_pins.append(pin_id)
        
        nets.append(Net(
            name=net_name, index=net_idx, pins=tuple(net_pins),
            net_class="Default", width_nm=250_000, clearance_nm=200_000,
            layer_mask=("L0", "L1", "L2", "L3"), criticality=0.3, criticality_source="default",
            kind=kind
        ))
        
        # Component
        comp = Component(
            ref=f"U{net_idx}", footprint="", value="", side=Side.TOP,
            position=Pt(mm_to_nm(rng.uniform(5, board_w-5)), mm_to_nm(rng.uniform(5, board_h-5))),
            rotation_deg=0.0, pins=tuple(p for p in pins if p.net == net_name),
            keepouts=(), locked=False, properties={}, source_id=""
        )
        components.append(comp)
    
    # Default rules
    rules = Rules(
        min_track_nm=100_000, min_clearance_nm=200_000,
        min_via_diameter_nm=400_000, min_via_drill_nm=200_000,
        min_annulus_nm=100_000,
        net_classes=(NetClass("Default", 250_000, 200_000, 600_000, 300_000),),
        via_types=(ViaType("Default", 600_000, 300_000, "L0", "L3"),),
        default_via="Default", defaults_assumed=True
    )
    
    copper = Copper(segments=(), arcs=(), vias=(), zones=())
    
    return Design(
        board=board, components=tuple(components), nets=tuple(nets),
        copper=copper, rules=rules, name=f"synth_{board_idx}"
    )


def _save_training_data(data: list[tuple[NetFeatures, NetRoutingLabel]], output_dir: str) -> None:
    """Save training data to JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    features_list = []
    labels_list = []
    
    for f, l in data:
        features_list.append({
            "net_name": f.net_name,
            "num_pins": f.num_pins,
            "bbox_width_mm": f.bbox_width_mm,
            "bbox_height_mm": f.bbox_height_mm,
            "hpwl_mm": f.hpwl_mm,
            "pin_density": f.pin_density,
            "num_layers": f.num_layers,
            "net_kind": f.net_kind,
            "width_mm": f.width_mm,
            "clearance_mm": f.clearance_mm,
            "has_existing_copper": f.has_existing_copper,
            "estimated_via_count": f.estimated_via_count,
            "escape_directions": f.escape_directions,
            "competing_nets": f.competing_nets,
            "local_obstacle_density": f.local_obstacle_density,
            "free_cell_ratio": f.free_cell_ratio,
            "estimated_congestion": f.estimated_congestion,
        })
        labels_list.append({
            "net_name": l.net_name,
            "route_success": l.route_success,
            "path_length_mm": l.path_length_mm,
            "via_count": l.via_count,
            "steps_taken": l.steps_taken,
            "status": l.status,
            "congestion_score": l.congestion_score,
            "max_congestion": l.max_congestion,
            "congestion_cells": l.congestion_cells,
        })
    
    with open(Path(output_dir) / "features.json", "w") as f:
        json.dump(features_list, f, indent=2)
    with open(Path(output_dir) / "labels.json", "w") as f:
        json.dump(labels_list, f, indent=2)


def load_training_data(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load training data from JSON files. Returns (X, y_difficulty, y_congestion)."""
    with open(Path(data_dir) / "features.json") as f:
        features_list = json.load(f)
    with open(Path(data_dir) / "labels.json") as f:
        labels_list = json.load(f)
    
    X = np.array([[f["num_pins"], f["bbox_width_mm"], f["bbox_height_mm"], f["hpwl_mm"],
                   f["pin_density"], f["num_layers"], f["net_kind"], f["width_mm"],
                   f["clearance_mm"], float(f["has_existing_copper"]), f["estimated_via_count"],
                   f["escape_directions"], f["competing_nets"], f["local_obstacle_density"],
                   f["free_cell_ratio"], f["estimated_congestion"], 1.0] for f in features_list], dtype=np.float32)
    
    # Target 1: routing difficulty (0=easy, 1=hard)
    # Use path_length + via_count*10 + (1-success)*100 as difficulty score
    y_difficulty = np.array([
        (0.0 if l["route_success"] else 100.0) + l["path_length_mm"] + l["via_count"] * 10.0
        for l in labels_list
    ], dtype=np.float32)
    
    # Target 2: congestion score (0-1)
    y_congestion = np.array([l["congestion_score"] for l in labels_list], dtype=np.float32)
    
    return X, y_difficulty, y_congestion


def train_net_ordering_model(
    data_dir: str = "ml_training_data",
    output_path: str = "models/router.pt",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train the net ordering MLP model."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install with: pip install -e '.[ml]'")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load data
    X, y_difficulty, y_congestion = load_training_data(data_dir)
    
    # Use difficulty target for net ordering
    y = y_difficulty
    
    # Split
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])
    
    # Normalize
    mean = X_train.mean(0)
    std = X_train.std(0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    
    # Model
    model = NetOrderingMLP(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train[idx], y_train[idx]
            
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(xb)
        
        epoch_loss /= len(X_train)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        val_losses.append(val_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "input_dim": X.shape[1],
    }, output_path)
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "model_path": output_path,
    }


def train_congestion_model(
    data_dir: str = "ml_training_data",
    output_path: str = "models/congestion.pt",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train the congestion prediction MLP model."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install with: pip install -e '.[ml]'")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load data
    X, y_difficulty, y_congestion = load_training_data(data_dir)
    
    # Split
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train = torch.from_numpy(X[train_idx])
    y_cong_train = torch.from_numpy(y_congestion[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_cong_val = torch.from_numpy(y_congestion[val_idx])
    
    # Normalize (use same mean/std as difficulty model or compute new ones)
    mean = X_train.mean(0)
    std = X_train.std(0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    
    # Model
    model = CongestionPredictionMLP(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train[idx], y_cong_train[idx]
            
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(xb)
        
        epoch_loss /= len(X_train)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_cong_val).item()
        val_losses.append(val_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "input_dim": X.shape[1],
    }, output_path)
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "model_path": output_path,
    }


def predict_net_order(design: Design, model_path: str | None = None) -> list[str]:
    """Predict net ordering using ML model, or fall back to heuristic."""
    if not TORCH_AVAILABLE or model_path is None or not Path(model_path).exists():
        # Heuristic fallback: sort by HPWL descending (route long nets first)
        nets_with_hpwl = []
        for net in design.nets:
            if len(net.pins) < 2:
                continue
            pins = [p for p in design.pins() if p.net == net.name]
            xs = [p.position.x for p in pins]
            ys = [p.position.y for p in pins]
            hpwl = (max(xs) - min(xs) + max(ys) - min(ys)) / 1_000_000.0
            nets_with_hpwl.append((net.name, hpwl))
        nets_with_hpwl.sort(key=lambda x: -x[1])
        return [n for n, _ in nets_with_hpwl]
    
    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = NetOrderingMLP(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    
    # Extract features for all routable nets
    kernel = GeometryKernel(design, KernelConfig())
    net_features = []
    for net in design.nets:
        if len(net.pins) < 2:
            continue
        f = extract_net_features(design, net, kernel)
        arr = features_to_array(f)
        arr_norm = (arr - mean.numpy()) / std.numpy()
        net_features.append((net.name, arr_norm))
    
    # Predict difficulty
    with torch.no_grad():
        X = torch.from_numpy(np.stack([arr for _, arr in net_features]))
        scores = model(X).numpy()
    
    # Sort by predicted difficulty (higher = harder, route first)
    net_features_with_scores = list(zip([n for n, _ in net_features], scores))
    net_features_with_scores.sort(key=lambda x: -x[1])
    
    return [n for n, _ in net_features_with_scores]


def predict_congestion(design: Design, model_path: str | None = None) -> dict[str, float]:
    """Predict congestion for all routable nets using ML model, or fall back to heuristic."""
    if not TORCH_AVAILABLE or model_path is None or not Path(model_path).exists():
        # Heuristic fallback: use estimated_congestion from features
        kernel = GeometryKernel(design, KernelConfig())
        congestion = {}
        for net in design.nets:
            if len(net.pins) < 2:
                continue
            f = extract_net_features(design, net, kernel)
            congestion[net.name] = f.estimated_congestion
        return congestion
    
    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = CongestionPredictionMLP(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    
    # Extract features for all routable nets
    kernel = GeometryKernel(design, KernelConfig())
    net_features = []
    for net in design.nets:
        if len(net.pins) < 2:
            continue
        f = extract_net_features(design, net, kernel)
        arr = features_to_array(f)
        arr_norm = (arr - mean.numpy()) / std.numpy()
        net_features.append((net.name, arr_norm))
    
    # Predict congestion
    with torch.no_grad():
        X = torch.from_numpy(np.stack([arr for _, arr in net_features]))
        scores = model(X).numpy()
    
    congestion = {}
    for (net_name, _), score in zip(net_features, scores):
        congestion[net_name] = max(0.0, min(1.0, float(score)))
    
    return congestion


def evaluate_ordering(design: Design, heuristic_order: list[str], ml_order: list[str]) -> dict:
    """Compare heuristic vs ML net ordering."""
    config = RouterConfig(grid_nm=500_000, max_steps_per_net=10000)
    
    # Test heuristic ordering
    router_h = GridRouter(design, config)
    results_h, metrics_h = router_h.route_all(heuristic_order)
    
    # Test ML ordering (new router instance)
    router_m = GridRouter(design, config)
    results_m, metrics_m = router_m.route_all(ml_order)
    
    return {
        "heuristic": {
            "routed": metrics_h.nets_routed,
            "partial": metrics_h.nets_partial,
            "failed": metrics_h.nets_failed,
            "total_length_mm": metrics_h.total_wire_length_mm,
            "total_vias": metrics_h.total_vias,
            "time_s": metrics_h.total_time_s,
        },
        "ml": {
            "routed": metrics_m.nets_routed,
            "partial": metrics_m.nets_partial,
            "failed": metrics_m.nets_failed,
            "total_length_mm": metrics_m.total_wire_length_mm,
            "total_vias": metrics_m.total_vias,
            "time_s": metrics_m.total_time_s,
        },
        "improvement": {
            "routed_delta": metrics_m.nets_routed - metrics_h.nets_routed,
            "length_delta_mm": metrics_m.total_wire_length_mm - metrics_h.total_wire_length_mm,
            "vias_delta": metrics_m.total_vias - metrics_h.total_vias,
        }
    }