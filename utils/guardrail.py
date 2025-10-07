# Standard library imports
import gc
import os
import warnings
from typing import List, Optional, Union

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

# Options
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None


# Guardrail class
class Guardrail:
    # Init
    def __init__(self):
        """
        Initialize the Guardrail class. This class provides methods to create
        smart ensemble predictions based on intermittence and coefficient of variation
        features to improve forecast quality.
        """
        pass

    def create_smart_ensemble(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        pred_col: str,
        baseline_col: str,
        intermittence_col: Optional[str] = None,
        cov_col: Optional[str] = None,
        intermittence_threshold: float = 50.0,
        cov_threshold: float = 1.0,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Create a smart ensemble prediction that intelligently combines model predictions
        and baselines based on intermittence and coefficient of variation features.
        
        The logic is:
        - If intermittence is high OR coefficient of variation is high → use baseline
        - Otherwise → use model prediction
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame containing predictions, baselines, and feature columns.
        group_cols : list
            List of columns to group by (e.g., ['client', 'warehouse', 'product']).
        pred_col : str
            Name of the prediction column (e.g., 'prediction').
        baseline_col : str
            Name of the baseline column (e.g., 'baseline_sales_ma_13').
        intermittence_col : str, optional
            Name of the intermittence feature column. If None, will auto-detect.
        cov_col : str, optional
            Name of the coefficient of variation feature column. If None, will auto-detect.
        intermittence_threshold : float, default=50.0
            Threshold for intermittence (percentage). Above this, use baseline.
        cov_threshold : float, default=1.0
            Threshold for coefficient of variation. Above this, use baseline.
        verbose : bool, default=True
            Whether to print detailed information during execution.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional 'prediction_ensemble' column.
        """
        if verbose:
            print("\n" + "=" * 70)
            print("SMART ENSEMBLE GUARDRAIL")
            print("=" * 70)
            print(f"Prediction column: '{pred_col}'")
            print(f"Baseline column: '{baseline_col}'")
            print(f"Group columns: {group_cols}")
            print(f"Intermittence threshold: {intermittence_threshold}%")
            print(f"CV threshold: {cov_threshold}")
        
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Auto-detect feature columns if not provided
        if intermittence_col is None:
            intermittence_col = self._find_feature_column(result_df, "intermittence")
            if intermittence_col is None:
                raise ValueError("Could not find intermittence feature column. Expected pattern: 'feature_*_intermittence'")
        if cov_col is None:
            cov_col = self._find_feature_column(result_df, "cov")
            if cov_col is None:
                raise ValueError("Could not find coefficient of variation feature column. Expected pattern: 'feature_*_cov'")
        if verbose:
            print(f"Using intermittence column: '{intermittence_col}'")
            print(f"Using CV column: '{cov_col}'")
        
        # Validate required columns exist
        required_cols = [pred_col, baseline_col, intermittence_col, cov_col] + group_cols
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Initialize the ensemble column
        result_df["prediction_ensemble"] = result_df[pred_col].copy()
        
        # Calculate ensemble logic for each group
        ensemble_stats = []
        
        # Loop by group
        for group_values, group_data in result_df.groupby(group_cols):
            # Get the first row of the group to check feature values
            first_row = group_data.iloc[0]
            intermittence_val = first_row[intermittence_col]
            cov_val = first_row[cov_col]
            
            # Determine if we should use baseline for this group
            use_baseline = (
                intermittence_val >= intermittence_threshold or 
                cov_val >= cov_threshold
            )
            
            # Apply the ensemble logic
            if use_baseline:
                # Use baseline for this group
                group_indices = group_data.index
                result_df.loc[group_indices, "prediction_ensemble"] = result_df.loc[group_indices, baseline_col]
                decision = "baseline"
            else:
                # Use prediction for this group
                group_indices = group_data.index
                result_df.loc[group_indices, "prediction_ensemble"] = result_df.loc[group_indices, pred_col]
                decision = "prediction"
            
            # Store statistics for reporting
            ensemble_stats.append({
                'group': group_values,
                'intermittence': intermittence_val,
                'cov': cov_val,
                'decision': decision,
                'rows_affected': len(group_data)
            })
        
        # Convert stats to DataFrame for analysis
        stats_df = pd.DataFrame(ensemble_stats)
        
        if verbose:
            # Print summary statistics
            baseline_groups = stats_df[stats_df['decision'] == 'baseline']
            prediction_groups = stats_df[stats_df['decision'] == 'prediction']
            
            print(f"\n📊 Ensemble Summary:")
            print(f"   • Total groups: {len(stats_df)}")
            print(f"   • Groups using baseline: {len(baseline_groups)} ({len(baseline_groups)/len(stats_df)*100:.1f}%)")
            print(f"   • Groups using prediction: {len(prediction_groups)} ({len(prediction_groups)/len(stats_df)*100:.1f}%)")
            
            if len(baseline_groups) > 0:
                print(f"\n🔍 Baseline Groups (high intermittence/CV):")
                print(f"   • Avg intermittence: {baseline_groups['intermittence'].mean():.1f}%")
                print(f"   • Avg CV: {baseline_groups['cov'].mean():.2f}")
                print(f"   • Rows affected: {baseline_groups['rows_affected'].sum()}")
            
            if len(prediction_groups) > 0:
                print(f"\n🎯 Prediction Groups (low intermittence/CV):")
                print(f"   • Avg intermittence: {prediction_groups['intermittence'].mean():.1f}%")
                print(f"   • Avg CV: {prediction_groups['cov'].mean():.2f}")
                print(f"   • Rows affected: {prediction_groups['rows_affected'].sum()}")
        
        # Store statistics for potential analysis
        self.ensemble_stats = stats_df
        
        if verbose:
            print(f"\n✅ Smart ensemble created successfully!")
            print(f"   • New column: 'prediction_ensemble'")
            print(f"   • Total rows processed: {len(result_df)}")
        
        return result_df
    
    def _find_feature_column(self, df: pd.DataFrame, feature_type: str) -> Optional[str]:
        """
        Helper function to find feature columns by pattern.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame to search in.
        feature_type : str
            Type of feature to find ('intermittence' or 'cov').
            
        Returns:
        --------
        str or None
            Name of the found column, or None if not found.
        """
        if feature_type == "intermittence":
            pattern = "feature_.*_intermittence"
        elif feature_type == "cov":
            pattern = "feature_.*_cov"
        else:
            return None
        
        import re
        matching_cols = [col for col in df.columns if re.match(pattern, col)]
        return matching_cols[0] if matching_cols else None
    
    def get_ensemble_stats(self) -> Optional[pd.DataFrame]:
        """
        Get the ensemble statistics from the last run.
        
        Returns:
        --------
        pd.DataFrame or None
            Statistics from the last ensemble creation, or None if not available.
        """
        return getattr(self, 'ensemble_stats', None)
    
    def plot_ensemble_decision(self, save_path: Optional[str] = None):
        """
        Create a visualization of the ensemble decision logic.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, displays the plot.
        """
        if not hasattr(self, 'ensemble_stats') or self.ensemble_stats is None:
            print("No ensemble statistics available. Run create_smart_ensemble first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Intermittence vs CV colored by decision
        scatter = ax1.scatter(
            self.ensemble_stats['intermittence'],
            self.ensemble_stats['cov'],
            c=self.ensemble_stats['decision'].map({'baseline': 'red', 'prediction': 'blue'}),
            alpha=0.7,
            s=50
        )
        ax1.set_xlabel('Intermittence (%)')
        ax1.set_ylabel('Coefficient of Variation')
        ax1.set_title('Ensemble Decision Logic')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Baseline'),
            Patch(facecolor='blue', label='Prediction')
        ]
        ax1.legend(handles=legend_elements)
        
        # Plot 2: Decision distribution
        decision_counts = self.ensemble_stats['decision'].value_counts()
        ax2.pie(decision_counts.values, labels=decision_counts.index, autopct='%1.1f%%')
        ax2.set_title('Decision Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
