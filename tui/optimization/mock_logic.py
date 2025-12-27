import random
import asyncio
import math
from typing import List, Dict
from tui.optimization.models import OptimizationProject, ParameterRanges, OptimizationResult

class MockOptimizer:
    """
    Simulates Bayesian Optimization logic.
    """
    
    def __init__(self, project: OptimizationProject):
        self.project = project
        
    def generate_initial_samples(self, count: int = 5) -> List[Dict[str, float]]:
        """
        Generate initial random samples (Latin Hypercube mock).
        """
        ranges = self.project.parameter_ranges
        samples = []
        
        for _ in range(count):
            params = {}
            if ranges.optimize_diameter:
                params['diameter'] = random.randint(ranges.diameter_min, ranges.diameter_max)
            else:
                params['diameter'] = 30 # Default
                
            if ranges.optimize_min_size:
                params['min_size'] = random.randint(ranges.min_size_min, ranges.min_size_max)
            else:
                params['min_size'] = 15
                
            if ranges.optimize_flow_threshold:
                params['flow_threshold'] = random.uniform(ranges.flow_threshold_min, ranges.flow_threshold_max)
            else:
                params['flow_threshold'] = 0.4
                
            if ranges.optimize_cellprob_threshold:
                params['cellprob_threshold'] = random.uniform(ranges.cellprob_threshold_min, ranges.cellprob_threshold_max)
            else:
                params['cellprob_threshold'] = 0.0
            
            samples.append(params)
            
        return samples

    def suggest_next_parameters(self) -> Dict[str, float]:
        """
        Suggest the next set of parameters based on history (Mock).
        Ideally this would use GP/acquisition function.
        For mock, we just perturb the best known parameters or explore.
        """
        ranges = self.project.parameter_ranges
        best_params = self.project.status.best_parameters
        
        # If no history, random sample
        if not best_params:
            return self.generate_initial_samples(1)[0]
            
        # 30% chance of random exploration, 70% chance of local optimization around best
        if random.random() < 0.3:
             return self.generate_initial_samples(1)[0]
        
        # Local optimization: perturb best params slightly
        new_params = best_params.copy()
        
        if ranges.optimize_diameter:
            delta = random.randint(-5, 5)
            new_params['diameter'] = max(ranges.diameter_min, min(ranges.diameter_max, new_params['diameter'] + delta))
            
        if ranges.optimize_min_size:
            delta = random.randint(-3, 3)
            new_params['min_size'] = max(ranges.min_size_min, min(ranges.min_size_max, new_params['min_size'] + delta))
            
        if ranges.optimize_flow_threshold:
            delta = random.uniform(-0.1, 0.1)
            new_params['flow_threshold'] = max(ranges.flow_threshold_min, min(ranges.flow_threshold_max, new_params['flow_threshold'] + delta))

        return new_params

    async def evaluate_parameters(self, params: Dict[str, float]) -> OptimizationResult:
        """
        Simulate running segmentation and calculating stats.
        Returns quality metrics based on how close parameters are to a 'hidden' optimal.
        """
        # Simulate processing time
        await asyncio.sleep(1.0) # 1 second delay
        
        # Define a "Target" optimal (hidden)
        target_diameter = 45
        target_min_size = 20
        target_flow = 0.4
        
        # Calculate distance-based quality (1.0 is perfect match)
        d_diff = abs(params.get('diameter', 30) - target_diameter) / 100.0
        m_diff = abs(params.get('min_size', 15) - target_min_size) / 50.0
        f_diff = abs(params.get('flow_threshold', 0.4) - target_flow) / 1.0
        
        # Add some noise
        noise = random.uniform(-0.05, 0.05)
        
        # Base quality - inverse of distance (used for mean_solidity as primary indicator)
        raw_quality = 1.0 - (d_diff + m_diff + f_diff) / 3.0 + noise
        quality = max(0.0, min(1.0, raw_quality))
        
        # Mock metrics
        num_cells = int(quality * 500 + random.randint(-50, 50))
        mean_area = float(params.get('diameter', 30)) ** 2 * 0.8  # correlated with diameter
        
        # Generate mock quality metrics
        mean_solidity = quality  # Primary quality indicator
        mean_circularity = 0.7 + quality * 0.2 + random.uniform(-0.1, 0.1)  # 0.7-0.9 range
        mean_circularity = max(0.0, min(1.0, mean_circularity))
        euler_integrity_pct = 85.0 + quality * 10.0 + random.uniform(-5, 5)  # 85-95% range
        euler_integrity_pct = max(0.0, min(100.0, euler_integrity_pct))
        fcr = 0.92 + quality * 0.06 + random.uniform(-0.03, 0.03)  # 0.92-0.98 range
        fcr = max(0.0, min(1.0, fcr))
        cv_area = 0.3 - quality * 0.1 + random.uniform(-0.05, 0.05)  # Lower is better
        cv_area = max(0.0, cv_area)
        geometric_cv = 0.25 - quality * 0.08 + random.uniform(-0.03, 0.03)
        geometric_cv = max(0.0, geometric_cv)
        lognormal_pvalue = 0.1 + quality * 0.8 + random.uniform(-0.1, 0.1)  # 0.1-0.9 range
        lognormal_pvalue = max(0.0, min(1.0, lognormal_pvalue))
        mean_eccentricity = 0.3 - quality * 0.15 + random.uniform(-0.05, 0.05)  # Lower is better
        mean_eccentricity = max(0.0, min(1.0, mean_eccentricity))
        
        return OptimizationResult(
            num_cells=num_cells,
            mean_area=round(mean_area, 2),
            mean_solidity=round(mean_solidity, 4),
            mean_circularity=round(mean_circularity, 4),
            euler_integrity_pct=round(euler_integrity_pct, 2),
            fcr=round(fcr, 4),
            cv_area=round(cv_area, 4),
            geometric_cv=round(geometric_cv, 4),
            lognormal_pvalue=round(lognormal_pvalue, 4),
            mean_eccentricity=round(mean_eccentricity, 4)
        )

