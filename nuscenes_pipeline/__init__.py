"""
nuScenes VQA Pipeline for Autonomous Driving Scene Understanding.

Pipeline stages:
  1A. risk_assessment    - Risk/hazard analysis per nuScenes sample
  1B. traffic_analysis   - Traffic signal state determination
  2.  question_selector  - Applicable question template selection
  3.  answer_generator   - Contrastive QA pair generation
"""

__version__ = "1.0.0"
