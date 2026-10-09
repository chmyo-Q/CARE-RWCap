"""Compatibility imports for models released under the former BPR name."""
from capr.model import (
    N, PosEmbed2D, DepthSepBlock, FaceSolver, DeepRWCapPredictor, ResidualHead,
    CAPRPredictor, _extract_state, load_deeprwcap, build_model,
    trainable_parameter_count, total_parameter_count,
    P0Predictor, ResidualFactorizedPredictor, load_p0,
)

BPRPredictor = CAPRPredictor
