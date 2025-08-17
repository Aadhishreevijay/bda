import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import random
from copy import deepcopy

sys.path.append('utils')
sys.path.append('pipeline')
sys.path.append('model')

import utils.misc as utils
from utils.default_config import set_default_config, get_choice_default_config
from pipeline.c_dataset_dataloader import CLDataSet
from model.consistent_predict_encoder import CLModel
from model.consistent_label_transformation import CLTransform


def load_trained_model(args, config):
    """Load the trained Con4m model"""
    print("Loading trained model...")
    
    # Construct the model
    cl_model = CLModel(config)
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.load_path, 'checkpoint_0.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return None
    
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load best model (based on F1 score)
    if "BestF1Model" in state_dict:
        cl_model.load_state_dict(state_dict["BestF1Model"], strict=False)
        print("Loaded best F1 model")
    else:
        cl_model.load_state_dict(state_dict["CLModel"], strict=False)
        print("Loaded last model")
    
    return cl_model


def test_specific_sample(model, dataset, device, sample_index):
    """Test a specific sample by index"""
    print(f"Testing specific sample at index {sample_index}...")
    
    # Get the dataset data directly
    data = dataset.data  # [num_level, seg_big_num, seg_small_num, n_features, length]
    label = dataset.label  # [num_level, seg_big_num, seg_small_num, n_class]
    
    # Check if index is valid
    if sample_index >= data.shape[1]:
        print(f"Error: Sample index {sample_index} is out of range. Max index is {data.shape[1]-1}")
        return None
    
    # Get the specific sample
    sample_data = data[:, sample_index, :, :, :]  # [num_level, seg_small_num, n_features, length]
    sample_labels = label[:, sample_index, :, :]  # [num_level, seg_small_num, n_class]
    
    # Use the first level for testing
    level_0_data = sample_data[0]  # [seg_small_num, n_features, length]
    level_0_labels = sample_labels[0]  # [seg_small_num, n_class]
    
    # Get ground truth (most common label)
    ground_truth = torch.argmax(level_0_labels, dim=1).mode()[0].item()
    
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS - SAMPLE {sample_index}")
    print(f"{'='*60}")
    print(f"Sample shape: {level_0_data.shape}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Number of segments: {level_0_data.shape[0]}")
    print(f"Features: {level_0_data.shape[1]}")
    print(f"Time points: {level_0_data.shape[2]}")
    
    # Prepare data for model
    batch_data = level_0_data.unsqueeze(0)  # [1, seg_small_num, n_features, length]
    batch_label = level_0_labels.unsqueeze(0)  # [1, seg_small_num, n_class]
    
    batch_data = batch_data.to(device)
    batch_label = batch_label.to(device)
    
    # Test the model
    with torch.no_grad():
        loss, hat_p, tilde_p, seg_y = model(x=batch_data, y=batch_label)
        
        # Get predictions for each segment
        segment_predictions = torch.argmax(hat_p, dim=-1)  # [1, seg_num]
        segment_confidences = torch.max(hat_p, dim=-1)[0]  # [1, seg_num]
        
        # Get final prediction (majority vote across segments)
        final_prediction = segment_predictions.mode()[0].item()
        final_confidence = segment_confidences.mean().item()
        
        # Get individual segment results
        segment_results = []
        for seg_idx in range(segment_predictions.shape[1]):
            seg_pred = segment_predictions[0, seg_idx].item()
            seg_conf = segment_confidences[0, seg_idx].item()
            seg_probs = hat_p[0, seg_idx]  # [n_class]
            segment_results.append({
                'segment': seg_idx,
                'prediction': seg_pred,
                'confidence': seg_conf,
                'prob_class_0': seg_probs[0].item(),
                'prob_class_1': seg_probs[1].item()
            })
    
    print(f"\nFinal Prediction: {final_prediction}")
    print(f"Final Confidence: {final_confidence:.3f}")
    print(f"Correct: {'✓' if final_prediction == ground_truth else '✗'}")
    
    # Show detailed segment-by-segment analysis
    print(f"\n{'='*60}")
    print("SEGMENT-BY-SEGMENT ANALYSIS")
    print(f"{'='*60}")
    print("Seg | Pred | Conf | P(Class0) | P(Class1) | Status")
    print("-" * 55)
    
    for seg_result in segment_results:
        status = "✓" if seg_result['prediction'] == ground_truth else "✗"
        print(f"{seg_result['segment']:3d} | {seg_result['prediction']:4d} | {seg_result['confidence']:.3f} | "
              f"{seg_result['prob_class_0']:.3f} | {seg_result['prob_class_1']:.3f} | {status}")
    
    # Calculate statistics
    correct_segments = sum(1 for r in segment_results if r['prediction'] == ground_truth)
    segment_accuracy = correct_segments / len(segment_results)
    
    print(f"\n{'='*60}")
    print("SAMPLE STATISTICS")
    print(f"{'='*60}")
    print(f"Segment Accuracy: {segment_accuracy:.3f} ({correct_segments}/{len(segment_results)} segments correct)")
    print(f"Average Confidence: {final_confidence:.3f}")
    print(f"Confidence Range: {min(r['confidence'] for r in segment_results):.3f} - {max(r['confidence'] for r in segment_results):.3f}")
    
    # Plot detailed timeline
    plot_detailed_timeline(segment_results, sample_index, ground_truth)
    
    return {
        'Sample_Index': sample_index,
        'Ground_Truth': ground_truth,
        'Predicted_Class': final_prediction,
        'Confidence': final_confidence,
        'Correct': final_prediction == ground_truth,
        'Segment_Accuracy': segment_accuracy,
        'Num_Segments': len(segment_results)
    }


def plot_detailed_timeline(segment_results, sample_index, ground_truth, save_dir='./specific_sample_results/'):
    """Plot detailed timeline for the specific sample"""
    os.makedirs(save_dir, exist_ok=True)
    
    segments = [r['segment'] for r in segment_results]
    predictions = [r['prediction'] for r in segment_results]
    confidences = [r['confidence'] for r in segment_results]
    prob_0 = [r['prob_class_0'] for r in segment_results]
    prob_1 = [r['prob_class_1'] for r in segment_results]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Predictions over time
    ax1.plot(segments, predictions, 'bo-', linewidth=2, markersize=6, label='Prediction')
    ax1.axhline(y=ground_truth, color='red', linestyle='--', alpha=0.7, label=f'Ground Truth: {ground_truth}')
    ax1.set_ylabel('Predicted Class')
    ax1.set_title(f'Sample {sample_index} - Predictions Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Plot 2: Confidence over time
    ax2.plot(segments, confidences, 'go-', linewidth=2, markersize=6, label='Confidence')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Threshold')
    ax2.set_ylabel('Confidence')
    ax2.set_title(f'Sample {sample_index} - Confidence Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Class probabilities over time
    ax3.plot(segments, prob_0, 'b-', linewidth=2, label='P(Class 0)', alpha=0.7)
    ax3.plot(segments, prob_1, 'r-', linewidth=2, label='P(Class 1)', alpha=0.7)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Boundary')
    ax3.set_xlabel('Segment')
    ax3.set_ylabel('Probability')
    ax3.set_title(f'Sample {sample_index} - Class Probabilities')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Confidence distribution
    ax4.hist(confidences, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(x=np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
    ax4.set_xlabel('Confidence')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Sample {sample_index} - Confidence Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'detailed_sample_{sample_index}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDetailed plots saved to: {save_dir}detailed_sample_{sample_index}.png")


def main():
    parser = argparse.ArgumentParser(description='Specific Sample Test for Con4m')
    parser.add_argument('--load_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--database_save_dir', type=str, required=True,
                        help='Path to the processed database')
    parser.add_argument('--data_name', type=str, default='fNIRS_2',
                        help='Dataset name')
    parser.add_argument('--noise_ratio', type=float, default=0.2,
                        help='Noise ratio')
    parser.add_argument('--exp_id', type=int, default=1,
                        help='Experiment ID')
    parser.add_argument('--sample_index', type=int, required=True,
                        help='Index of the specific sample to test')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    
    # Add missing arguments that the config function expects
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--window_time', type=float, default=1,
                        help='Window time')
    parser.add_argument('--slide_time', type=float, default=0.5,
                        help='Slide time')
    parser.add_argument('--num_level', type=int, default=5,
                        help='Number of levels')
    parser.add_argument('--n_process_loader', type=int, default=50,
                        help='Number of process loaders')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--d_inner', type=int, default=256,
                        help='Inner dimension')
    parser.add_argument('--kernel_size_list', nargs='*', type=int, default=[3, 3],
                        help='Kernel size list')
    parser.add_argument('--stride_size_list', nargs='*', type=int, default=[1, 1],
                        help='Stride size list')
    parser.add_argument('--padding_size_list', nargs='*', type=int, default=[1, 1],
                        help='Padding size list')
    parser.add_argument('--down_sampling', type=int, default=1,
                        help='Down sampling')
    parser.add_argument('--warm_epoch_num', type=int, default=10,
                        help='Warm epoch number')
    parser.add_argument('--cl_epoch_num', type=int, default=30,
                        help='CL epoch number')
    parser.add_argument('--level_gap_epoch_num', type=int, default=5,
                        help='Level gap epoch number')
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    args, config = get_choice_default_config(args)
    
    # Load test dataset
    print("Loading test dataset...")
    args.patient_list = args.test_patient_list
    test_dataset = CLDataSet(args)
    
    # Configure model with dataset information
    config.n_class = test_dataset.n_class
    config.seg_small_num = test_dataset.seg_small_num
    config.raw_input_len = test_dataset.data_handler.window_len
    config.n_features = test_dataset.n_features
    
    print(f"Model config - n_class: {config.n_class}, n_features: {config.n_features}")
    print(f"Window len: {config.raw_input_len}, seg_small_num: {config.seg_small_num}")
    print(f"Total samples available: {test_dataset.data.shape[1]}")
    
    # Load trained model
    model = load_trained_model(args, config)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    model.to(device)
    
    # Test specific sample
    result = test_specific_sample(model, test_dataset, device, args.sample_index)
    
    if result:
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Sample Index: {result['Sample_Index']}")
        print(f"Ground Truth: {result['Ground_Truth']}")
        print(f"Predicted Class: {result['Predicted_Class']}")
        print(f"Confidence: {result['Confidence']:.3f}")
        print(f"Correct: {'✓' if result['Correct'] else '✗'}")
        print(f"Segment Accuracy: {result['Segment_Accuracy']:.3f}")
        print(f"Number of Segments: {result['Num_Segments']}")


if __name__ == '__main__':
    main()
