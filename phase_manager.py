import logging

logger = logging.getLogger(__name__)


class PhaseManager:
    """
    Manages phased training with entropy coefficient annealing and threshold adjustments.
    """
    def __init__(self, total_phases=10, initial_entropy=0.05, final_entropy=0.0001,
                 initial_buy_threshold=0.15, final_buy_threshold=0.35,
                 initial_sell_threshold=-0.15, final_sell_threshold=-0.35):
        self.total_phases = total_phases
        self.initial_entropy = initial_entropy
        self.final_entropy = final_entropy
        self.initial_buy_threshold = initial_buy_threshold
        self.final_buy_threshold = final_buy_threshold
        self.initial_sell_threshold = initial_sell_threshold
        self.final_sell_threshold = final_sell_threshold
        self.current_phase = 1  # Start at phase 1

    def get_phase_params(self, phase):
        """Get parameters for a specific phase (1-indexed)."""
        if phase < 1 or phase > self.total_phases:
            raise ValueError(f"Phase must be between 1 and {self.total_phases}")

        if self.total_phases == 1:
            # Fallback for single phase training
            return {
                'entropy_coef': self.initial_entropy,
                'buy_threshold': 0.0,
                'sell_threshold': 0.0,
                'phase': phase
            }

        # Calculate interpolation factor
        progress = (phase - 1) / (self.total_phases - 1)  # 0.0 to 1.0

        # Anneal entropy coefficient
        entropy_coef = self.initial_entropy + (self.final_entropy - self.initial_entropy) * progress

        # Adjust thresholds (both buy and sell)
        buy_threshold = self.initial_buy_threshold + (self.final_buy_threshold - self.initial_buy_threshold) * progress
        sell_threshold = self.initial_sell_threshold + (self.final_sell_threshold - self.initial_sell_threshold) * progress

        return {
            'entropy_coef': entropy_coef,         # Use properly interpolated value
            'buy_threshold': buy_threshold,       # Interpolate from conservative to more permissive
            'sell_threshold': sell_threshold,     # Allows stronger signals in later phases
            'phase': phase
        }

    def get_steps_per_phase(self, total_timesteps):
        """Calculate steps per phase."""
        return total_timesteps // self.total_phases

    def log_phase_start(self, phase, steps_per_phase):
        """Log the start of a new phase."""
        params = self.get_phase_params(phase)
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING PHASE {phase}/{self.total_phases}")
        logger.info(f"Steps: {steps_per_phase:,}")
        logger.info(f"Entropy Coefficient: {params['entropy_coef']:.6f}")
        logger.info(f"Buy Threshold: {params['buy_threshold']:.3f}")
        logger.info(f"Sell Threshold: {params['sell_threshold']:.3f}")
        logger.info(f"{'='*60}\n")