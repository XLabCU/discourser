import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import warnings

logger = logging.getLogger(__name__)

class BaselineAnalysisEngine:
    """
    Statistical baseline analysis for influence detection.
    Generates null distributions and assesses significance of observed similarities.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Cache for baseline distributions to avoid recalculation
        self.baseline_cache = {}
        
        # Statistical thresholds
        self.significance_levels = {
            'very_high': 0.001,
            'high': 0.01,
            'moderate': 0.05,
            'weak': 0.1
        }
        
        # Effect size thresholds (Cohen's conventions, adapted)
        self.effect_size_thresholds = {
            'negligible': 0.1,
            'small': 0.3,
            'medium': 0.5,
            'large': 0.8,
            'very_large': 1.2
        }
    
    def generate_document_shuffle_baseline(self, 
                                         core_embeddings: np.ndarray,
                                         target_embeddings: np.ndarray,
                                         n_permutations: int = 1000,
                                         cache_key: str = None) -> Dict:
        """
        Generate baseline distribution by shuffling document pairings.
        
        Args:
            core_embeddings: Core corpus embeddings
            target_embeddings: Target corpus embeddings  
            n_permutations: Number of random permutations
            cache_key: Optional key for caching results
            
        Returns:
            Dictionary with baseline statistics
        """
        try:
            # Check cache first
            if cache_key and cache_key in self.baseline_cache:
                logger.info(f"Using cached baseline for {cache_key}")
                return self.baseline_cache[cache_key]
            
            logger.info(f"Generating document shuffle baseline with {n_permutations} permutations")
            
            # Calculate original similarity matrix for comparison
            original_similarity = cosine_similarity(target_embeddings, core_embeddings)
            original_mean = np.mean(original_similarity)
            
            # Generate permuted similarities
            permuted_means = []
            permuted_maxes = []
            permuted_distributions = []
            
            for i in range(n_permutations):
                if i % 100 == 0:
                    logger.debug(f"Permutation {i}/{n_permutations}")
                
                # Shuffle target embeddings
                shuffled_indices = np.random.permutation(len(target_embeddings))
                shuffled_target = target_embeddings[shuffled_indices]
                
                # Calculate similarity with shuffled targets
                shuffled_similarity = cosine_similarity(shuffled_target, core_embeddings)
                
                # Store statistics
                permuted_means.append(np.mean(shuffled_similarity))
                permuted_maxes.append(np.max(shuffled_similarity))
                permuted_distributions.append(shuffled_similarity.flatten())
            
            # Compile baseline statistics
            baseline_stats = {
                'type': 'document_shuffle',
                'n_permutations': n_permutations,
                'original_mean_similarity': float(original_mean),
                'baseline_mean': float(np.mean(permuted_means)),
                'baseline_std': float(np.std(permuted_means)),
                'baseline_distribution': np.array(permuted_means),
                'baseline_max_distribution': np.array(permuted_maxes),
                'full_distributions': permuted_distributions,
                'percentiles': {
                    p: float(np.percentile(permuted_means, p)) 
                    for p in [5, 10, 25, 50, 75, 90, 95, 99]
                }
            }
            
            # Cache if key provided
            if cache_key:
                self.baseline_cache[cache_key] = baseline_stats
            
            logger.info(f"Baseline generation complete. Mean baseline: {baseline_stats['baseline_mean']:.4f}")
            return baseline_stats
            
        except Exception as e:
            logger.error(f"Error generating document shuffle baseline: {str(e)}")
            return {}
    
    def generate_vector_projection_baseline(self,
                                          embeddings: np.ndarray,
                                          custom_vector: np.ndarray,
                                          n_permutations: int = 1000) -> Dict:
        """
        Generate baseline for vector projection analysis by projecting onto random vectors.
        
        Args:
            embeddings: Document embeddings
            custom_vector: The custom vector being analyzed
            n_permutations: Number of random vectors to generate
            
        Returns:
            Dictionary with baseline projection statistics
        """
        try:
            logger.info(f"Generating vector projection baseline with {n_permutations} random vectors")
            
            # Calculate original projections
            normalized_vector = custom_vector / np.linalg.norm(custom_vector)
            original_projections = np.dot(embeddings, normalized_vector)
            original_variance = np.var(original_projections)
            original_range = np.max(original_projections) - np.min(original_projections)
            
            # Generate random vector projections
            random_variances = []
            random_ranges = []
            random_means = []
            
            for i in range(n_permutations):
                # Generate random unit vector
                random_vector = np.random.randn(len(custom_vector))
                random_vector = random_vector / np.linalg.norm(random_vector)
                
                # Project embeddings onto random vector
                random_projections = np.dot(embeddings, random_vector)
                
                random_variances.append(np.var(random_projections))
                random_ranges.append(np.max(random_projections) - np.min(random_projections))
                random_means.append(np.mean(np.abs(random_projections)))
            
            baseline_stats = {
                'type': 'vector_projection',
                'n_permutations': n_permutations,
                'original_variance': float(original_variance),
                'original_range': float(original_range),
                'baseline_variance_mean': float(np.mean(random_variances)),
                'baseline_variance_std': float(np.std(random_variances)),
                'baseline_range_mean': float(np.mean(random_ranges)),
                'baseline_range_std': float(np.std(random_ranges)),
                'variance_distribution': np.array(random_variances),
                'range_distribution': np.array(random_ranges)
            }
            
            return baseline_stats
            
        except Exception as e:
            logger.error(f"Error generating vector projection baseline: {str(e)}")
            return {}
    
    def generate_corpus_overlap_baseline(self,
                                       core_embeddings: np.ndarray,
                                       target_embeddings: np.ndarray,
                                       custom_vector: np.ndarray,
                                       n_permutations: int = 500) -> Dict:
        """
        Generate baseline for corpus overlap analysis by shuffling corpus labels.
        
        Args:
            core_embeddings: Core corpus embeddings
            target_embeddings: Target corpus embeddings
            custom_vector: Vector for projection analysis
            n_permutations: Number of label shuffles
            
        Returns:
            Dictionary with baseline overlap statistics
        """
        try:
            logger.info(f"Generating corpus overlap baseline with {n_permutations} permutations")
            
            # Combine embeddings and create labels
            combined_embeddings = np.vstack([core_embeddings, target_embeddings])
            true_labels = np.array(['core'] * len(core_embeddings) + ['target'] * len(target_embeddings))
            
            # Calculate original overlap metrics
            normalized_vector = custom_vector / np.linalg.norm(custom_vector)
            original_projections = np.dot(combined_embeddings, normalized_vector)
            
            core_projections = original_projections[:len(core_embeddings)]
            target_projections = original_projections[len(core_embeddings):]
            
            original_overlap = self._calculate_distribution_overlap(core_projections, target_projections)
            
            # Generate permuted overlaps
            permuted_overlaps = []
            permuted_mean_diffs = []
            permuted_cohens_d = []
            
            for i in range(n_permutations):
                # Shuffle labels
                shuffled_labels = np.random.permutation(true_labels)
                
                # Split by shuffled labels
                shuffled_core_mask = shuffled_labels == 'core'
                shuffled_target_mask = shuffled_labels == 'target'
                
                shuffled_core_proj = original_projections[shuffled_core_mask]
                shuffled_target_proj = original_projections[shuffled_target_mask]
                
                # Calculate overlap metrics
                overlap = self._calculate_distribution_overlap(shuffled_core_proj, shuffled_target_proj)
                mean_diff = np.mean(shuffled_core_proj) - np.mean(shuffled_target_proj)
                
                # Cohen's d
                pooled_std = np.sqrt((np.var(shuffled_core_proj) + np.var(shuffled_target_proj)) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                permuted_overlaps.append(overlap)
                permuted_mean_diffs.append(mean_diff)
                permuted_cohens_d.append(abs(cohens_d))
            
            baseline_stats = {
                'type': 'corpus_overlap',
                'n_permutations': n_permutations,
                'original_overlap': original_overlap,
                'baseline_overlap_mean': float(np.mean(permuted_overlaps)),
                'baseline_overlap_std': float(np.std(permuted_overlaps)),
                'baseline_mean_diff_mean': float(np.mean(permuted_mean_diffs)),
                'baseline_mean_diff_std': float(np.std(permuted_mean_diffs)),
                'baseline_cohens_d_mean': float(np.mean(permuted_cohens_d)),
                'overlap_distribution': np.array(permuted_overlaps),
                'mean_diff_distribution': np.array(permuted_mean_diffs),
                'cohens_d_distribution': np.array(permuted_cohens_d)
            }
            
            return baseline_stats
            
        except Exception as e:
            logger.error(f"Error generating corpus overlap baseline: {str(e)}")
            return {}
    
    def _calculate_distribution_overlap(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """Calculate overlap between two distributions using histogram method"""
        try:
            # Create common bins
            all_values = np.concatenate([dist1, dist2])
            bins = np.linspace(np.min(all_values), np.max(all_values), 50)
            
            # Calculate histograms
            hist1, _ = np.histogram(dist1, bins=bins, density=True)
            hist2, _ = np.histogram(dist2, bins=bins, density=True)
            
            # Calculate overlap (minimum of the two histograms)
            overlap = np.sum(np.minimum(hist1, hist2)) * (bins[1] - bins[0])
            
            return float(overlap)
            
        except Exception:
            return 0.0
    
    def assess_significance(self, 
                           observed_value: float,
                           baseline_distribution: np.ndarray,
                           test_type: str = 'two_tailed') -> Dict:
        """
        Assess statistical significance of observed value against baseline.
        
        Args:
            observed_value: The observed statistic
            baseline_distribution: Null distribution from permutations
            test_type: 'two_tailed', 'greater', or 'less'
            
        Returns:
            Dictionary with significance statistics
        """
        try:
            baseline_mean = np.mean(baseline_distribution)
            baseline_std = np.std(baseline_distribution)
            
            # Calculate p-value based on test type
            if test_type == 'greater':
                p_value = np.mean(baseline_distribution >= observed_value)
            elif test_type == 'less':
                p_value = np.mean(baseline_distribution <= observed_value)
            else:  # two_tailed
                # Distance from mean
                observed_distance = abs(observed_value - baseline_mean)
                baseline_distances = np.abs(baseline_distribution - baseline_mean)
                p_value = np.mean(baseline_distances >= observed_distance)
            
            # Calculate z-score and effect size
            z_score = (observed_value - baseline_mean) / baseline_std if baseline_std > 0 else 0
            effect_size = z_score  # Standardized effect size
            
            # Determine significance level
            significance_level = self._categorize_significance(p_value)
            effect_size_category = self._categorize_effect_size(abs(effect_size))
            
            return {
                'observed_value': float(observed_value),
                'baseline_mean': float(baseline_mean),
                'baseline_std': float(baseline_std),
                'p_value': float(p_value),
                'z_score': float(z_score),
                'effect_size': float(effect_size),
                'significance_level': significance_level,
                'effect_size_category': effect_size_category,
                'interpretation': self._generate_interpretation(p_value, effect_size, significance_level, effect_size_category),
                'confidence_interval': self._calculate_confidence_interval(baseline_distribution)
            }
            
        except Exception as e:
            logger.error(f"Error assessing significance: {str(e)}")
            return {'error': str(e)}
    
    def _categorize_significance(self, p_value: float) -> str:
        """Categorize p-value into significance levels"""
        if p_value <= self.significance_levels['very_high']:
            return 'very_high'
        elif p_value <= self.significance_levels['high']:
            return 'high'
        elif p_value <= self.significance_levels['moderate']:
            return 'moderate'
        elif p_value <= self.significance_levels['weak']:
            return 'weak'
        else:
            return 'non_significant'
    
    def _categorize_effect_size(self, effect_size: float) -> str:
        """Categorize effect size magnitude"""
        if effect_size >= self.effect_size_thresholds['very_large']:
            return 'very_large'
        elif effect_size >= self.effect_size_thresholds['large']:
            return 'large'
        elif effect_size >= self.effect_size_thresholds['medium']:
            return 'medium'
        elif effect_size >= self.effect_size_thresholds['small']:
            return 'small'
        else:
            return 'negligible'
    
    def _generate_interpretation(self, p_value: float, effect_size: float, 
                               significance_level: str, effect_size_category: str) -> str:
        """Generate human-readable interpretation"""
        
        # Base interpretation on significance and effect size
        if significance_level == 'non_significant':
            if effect_size_category in ['negligible', 'small']:
                return "No evidence of influence above chance levels. Observed similarity is consistent with random variation."
            else:
                return "Large effect observed but not statistically significant. May indicate genuine influence but requires more data for confirmation."
        
        # Significant results
        confidence_map = {
            'very_high': 'very strong',
            'high': 'strong', 
            'moderate': 'moderate',
            'weak': 'weak'
        }
        
        confidence = confidence_map.get(significance_level, 'weak')
        
        if effect_size_category in ['large', 'very_large']:
            return f"Strong evidence of influence. {confidence.title()} statistical significance with large practical effect."
        elif effect_size_category == 'medium':
            return f"Moderate evidence of influence. {confidence.title()} statistical significance with medium practical effect."
        elif effect_size_category == 'small':
            return f"Weak evidence of influence. {confidence.title()} statistical significance but small practical effect."
        else:
            return f"Minimal evidence of influence. Statistically significant but negligible practical effect."
    
    def _calculate_confidence_interval(self, distribution: np.ndarray, 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for baseline distribution"""
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        return (
            float(np.percentile(distribution, lower_percentile)),
            float(np.percentile(distribution, upper_percentile))
        )
    
    def compare_multiple_analyses(self, 
                                observations: List[float],
                                baseline_distributions: List[np.ndarray],
                                labels: List[str],
                                correction_method: str = 'bonferroni') -> Dict:
        """
        Compare multiple analyses with multiple comparison correction.
        
        Args:
            observations: List of observed values
            baseline_distributions: List of corresponding baseline distributions
            labels: Labels for each analysis
            correction_method: 'bonferroni', 'fdr', or 'none'
            
        Returns:
            Dictionary with corrected significance results
        """
        try:
            # Calculate raw p-values
            raw_p_values = []
            effect_sizes = []
            
            for obs, baseline in zip(observations, baseline_distributions):
                result = self.assess_significance(obs, baseline)
                raw_p_values.append(result['p_value'])
                effect_sizes.append(result['effect_size'])
            
            # Apply correction
            if correction_method == 'bonferroni':
                corrected_p_values = [p * len(raw_p_values) for p in raw_p_values]
                corrected_p_values = [min(p, 1.0) for p in corrected_p_values]  # Cap at 1.0
            elif correction_method == 'fdr':
                corrected_p_values = self._fdr_correction(raw_p_values)
            else:
                corrected_p_values = raw_p_values
            
            # Compile results
            results = []
            for i, (label, obs, raw_p, corr_p, effect) in enumerate(
                zip(labels, observations, raw_p_values, corrected_p_values, effect_sizes)
            ):
                results.append({
                    'label': label,
                    'observed_value': obs,
                    'raw_p_value': raw_p,
                    'corrected_p_value': corr_p,
                    'effect_size': effect,
                    'significant_after_correction': corr_p <= 0.05,
                    'significance_level': self._categorize_significance(corr_p)
                })
            
            return {
                'correction_method': correction_method,
                'n_comparisons': len(observations),
                'results': results,
                'summary': self._summarize_multiple_comparisons(results)
            }
            
        except Exception as e:
            logger.error(f"Error in multiple comparisons: {str(e)}")
            return {'error': str(e)}
    
    def _fdr_correction(self, p_values: List[float], alpha: float = 0.05) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction"""
        p_array = np.array(p_values)
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]
        
        # Calculate FDR critical values
        m = len(p_values)
        critical_values = [(i + 1) / m * alpha for i in range(m)]
        
        # Find largest i where p[i] <= critical_value[i]
        significant_mask = sorted_p <= critical_values
        
        if np.any(significant_mask):
            max_significant_idx = np.where(significant_mask)[0][-1]
            corrected_p = sorted_p.copy()
            corrected_p[max_significant_idx + 1:] = 1.0  # Non-significant
        else:
            corrected_p = np.ones_like(sorted_p)
        
        # Restore original order
        restored_p = np.empty_like(corrected_p)
        restored_p[sorted_indices] = corrected_p
        
        return restored_p.tolist()
    
    def _summarize_multiple_comparisons(self, results: List[Dict]) -> Dict:
        """Summarize results from multiple comparisons"""
        total = len(results)
        significant = sum(1 for r in results if r['significant_after_correction'])
        
        return {
            'total_comparisons': total,
            'significant_after_correction': significant,
            'proportion_significant': significant / total if total > 0 else 0,
            'strongest_evidence': max(results, key=lambda x: abs(x['effect_size'])) if results else None,
            'most_significant': min(results, key=lambda x: x['corrected_p_value']) if results else None
        }
    
    def export_baseline_results(self, baseline_stats: Dict, filename: str = None) -> str:
        """Export baseline analysis results to JSON for reproducibility"""
        try:
            import json
            from datetime import datetime
            
            # Prepare export data (convert numpy arrays to lists)
            export_data = baseline_stats.copy()
            
            # Convert numpy arrays to lists for JSON serialization
            for key, value in export_data.items():
                if isinstance(value, np.ndarray):
                    export_data[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    export_data[key] = [arr.tolist() for arr in value]
            
            # Add metadata
            export_data['exported_at'] = datetime.now().isoformat()
            export_data['random_seed'] = self.random_seed
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"baseline_analysis_{timestamp}.json"
            
            # Export to JSON
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Baseline results exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting baseline results: {str(e)}")
            return ""