"""
Data Acquisition Module
Professional-grade multispectral data acquisition system
"""

from core.acquisition_manager import (
    DataAcquisitionManager,
    QualityValidator,
    CaptureMetadata,
    CaptureStatus
)

from core.capture_scheduler import (
    AutomaticCaptureScheduler,
    CaptureSchedule,
    CaptureMode
)

__version__ = '1.0.0'
__author__ = 'Agricultural Team'

__all__ = [
    'DataAcquisitionManager',
    'QualityValidator',
    'CaptureMetadata',
    'CaptureStatus',
    'AutomaticCaptureScheduler',
    'CaptureSchedule',
    'CaptureMode'
]
