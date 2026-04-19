"""Video processing utilities for extracting frames from videos."""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json
from tqdm import tqdm


class VideoProcessor:
    """Process videos to extract frames with specific sampling strategy."""
    
    def __init__(self, segment_duration: int = 5, frame_interval: int = 5, fps: int = 30):
        """
        Initialize video processor.
        
        Args:
            segment_duration: Duration of video segment in seconds
            frame_interval: Sample every Nth frame (e.g., 5 = every 5th frame)
            fps: Assumed frame rate (frames per second)
        """
        self.segment_duration = segment_duration
        self.frame_interval = frame_interval
        self.fps = fps
        self.frames_per_segment = segment_duration * fps
        self.sampled_frames_per_segment = self.frames_per_segment // frame_interval
    
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract frames from a video using the specified sampling strategy.
        
        Args:
            video_path: Path to input video file
            output_dir: Optional directory to save frames
            
        Returns:
            Tuple of (frames_list, frame_indices)
            - frames_list: List of frame arrays
            - frame_indices: List of original frame indices
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get actual FPS
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            self.fps = int(actual_fps)
            self.frames_per_segment = self.segment_duration * self.fps
            self.sampled_frames_per_segment = self.frames_per_segment // self.frame_interval
        
        frames = []
        frame_indices = []
        frame_count = 0
        segment_frame_count = 0
        
        # Extract first 5 seconds (or entire video if shorter)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we've reached segment duration
            if segment_frame_count >= self.frames_per_segment:
                break
            
            # Sample every Nth frame
            if frame_count % self.frame_interval == 0:
                frame = cv2.resize(frame, (112, 112))
                frames.append(frame.copy())
                frame_indices.append(frame_count)
                segment_frame_count += self.frame_interval
            
            frame_count += 1
        
        cap.release()
        
        # Save frames if output directory specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            video_name = video_path.stem
            for idx, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
                frame_path = output_dir / f"{video_name}_frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
        
        return frames, frame_indices
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    def process_video_batch(self, video_paths: List[str], output_base_dir: str, 
                           save_frames: bool = True) -> dict:
        """
        Process multiple videos in batch.
        
        Args:
            video_paths: List of video file paths
            output_base_dir: Base directory for output
            save_frames: Whether to save frames to disk
            
        Returns:
            Dictionary with processing results
        """
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'processed': [],
            'failed': [],
            'metadata': {}
        }
        
        for video_path in tqdm(video_paths, desc="Processing videos"):
            try:
                video_path = Path(video_path)
                video_name = video_path.stem
                
                # Get video info
                info = self.get_video_info(str(video_path))
                
                # Extract frames
                output_dir = output_base_dir / video_name if save_frames else None
                frames, frame_indices = self.extract_frames(str(video_path), output_dir)
                
                # Save metadata
                metadata = {
                    'video_name': video_name,
                    'video_path': str(video_path),
                    'num_frames': len(frames),
                    'frame_indices': frame_indices,
                    'video_info': info,
                    'segment_duration': self.segment_duration,
                    'frame_interval': self.frame_interval
                }
                
                metadata_path = output_base_dir / f"{video_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                results['processed'].append(video_name)
                results['metadata'][video_name] = metadata
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results['failed'].append(str(video_path))
        
        # Save summary
        summary_path = output_base_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

