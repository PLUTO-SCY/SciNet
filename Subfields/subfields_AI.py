comprehensive_ai_fields = [
    # =========================================================================
    # I. Core Machine Learning Paradigms
    # =========================================================================
    "Reinforcement Learning (RL)",
    "Deep Reinforcement Learning (DRL)",
    "Bandits / Multi-Armed Bandits",
    "Imitation Learning / Behavioral Cloning",
    "Inverse Reinforcement Learning",
    "Self-Supervised Learning (SSL)",
    "Contrastive Learning",
    "Masked Image Modeling",
    "Supervised Learning",
    "Semi-Supervised Learning",
    "Weakly-Supervised Learning",
    "Unsupervised Learning",
    "Federated Learning",
    "Continual Learning / Lifelong Learning",
    "Online Learning",
    "Multi-Task Learning",
    "Active Learning",
    "Transfer Learning",
    "Domain Adaptation & Generalization",
    "Meta-Learning (Learning to Learn)",
    "Few-Shot / Zero-Shot / One-Shot Learning",
    "Ensemble Learning",
    "Curriculum Learning",
    "Neuro-Symbolic AI",
    "Causal Representation Learning",
    "Human-in-the-Loop (HITL)",
    "Reinforcement Learning from Human Feedback (RLHF)",
    "Direct Preference Optimization (DPO)",

    # =========================================================================
    # II. Deep Learning Models, Architectures & Theory
    # =========================================================================
    # --- Generative Models ---
    "Generative Models",
    "Diffusion Models / Score-Based Generative Models",
    "Latent Diffusion Models",
    "Generative Adversarial Networks (GANs)",
    "Variational Autoencoders (VAEs)",
    "Flow-based Models (Normalizing Flows)",
    
    # --- Architectures ---
    "Transformers",
    "State Space Models (SSMs) / Mamba",
    "Recurrent Neural Networks (RNNs/LSTMs/GRUs)",
    "Graph Neural Networks (GNNs)",
    "Graph Attention Networks (GATs)",
    "Convolutional Neural Networks (CNNs)",
    "Spiking Neural Networks (SNNs)",
    "Liquid Neural Networks",
    "Neural Ordinary Differential Equations (Neural ODEs)",
    "Neural Operators (FNO/DeepONet)",
    "Capsule Networks",
    "Memory-Augmented Neural Networks",
    "Mixture of Experts (MoE)",
    
    # --- Theory ---
    "Deep Learning Theory",
    "Optimization Landscapes",
    "Generalization Theory",
    "Information Theory in DL",
    "Neural Tangent Kernels (NTK)",
    "Overparameterization & Double Descent",
    "Implicit Regularization",
    "Manifold Learning",
    "Spectral Graph Theory",

    # =========================================================================
    # III. Computer Vision
    # =========================================================================
    "Computer Vision",
    "Image Classification",
    "Object Detection",
    "Semantic Segmentation",
    "Instance & Panoptic Segmentation",
    "Face Recognition & Analysis",
    "Pose Estimation & Keypoint Detection",
    "Action Recognition & Video Understanding",
    "Video Generation & Editing",
    "Visual Tracking",
    "3D Computer Vision",
    "NeRF (Neural Radiance Fields) & Gaussian Splatting",
    "Point Cloud Processing",
    "Depth Estimation",
    "Optical Character Recognition (OCR)",
    "Document Layout Analysis",
    "Image Restoration (Denoising, Deblurring, Super-Resolution)",
    "Medical Image Analysis",
    "Biomedical Image Segmentation",
    "Remote Sensing & Satellite Imaging",
    "Vision-Language Models (VLMs)",
    "Visual Question Answering (VQA)",
    "Image Captioning",

    # =========================================================================
    # IV. Natural Language Processing (NLP)
    # =========================================================================
    "Natural Language Processing (NLP)",
    "Large Language Models (LLMs)",
    "Prompt Engineering",
    "In-Context Learning",
    "Chain-of-Thought (CoT) Reasoning",
    "Retrieval-Augmented Generation (RAG)",
    "Natural Language Generation (NLG)",
    "Machine Translation",
    "Text Summarization",
    "Question Answering (QA)",
    "Dialogue Systems & Chatbots",
    "Sentiment Analysis & Opinion Mining",
    "Named Entity Recognition (NER)",
    "Relation Extraction",
    "Knowledge Graphs (KG)",
    "Text Classification",
    "Information Retrieval (IR)",
    "Code Generation & Understanding (AI for Code)",
    "Automated Fact-Checking",

    # =========================================================================
    # V. Audio, Speech & Music
    # =========================================================================
    "Speech Recognition (ASR)",
    "Text-to-Speech (TTS) & Speech Synthesis",
    "Speaker Verification & Identification",
    "Audio Event Detection",
    "Music Generation & Analysis",
    "Voice Conversion & Cloning",
    "Sound Source Separation",
    "Audio-Visual Learning",

    # =========================================================================
    # VI. Robotics & Embodied AI
    # =========================================================================
    "Robotics",
    "Embodied AI",
    "Simultaneous Localization and Mapping (SLAM)",
    "Robot Manipulation & Grasping",
    "Motion Planning & Path Planning",
    "Human-Robot Interaction (HRI)",
    "Sim-to-Real Transfer",
    "Soft Robotics",
    "Swarm Robotics",
    "Autonomous Driving / Self-Driving Vehicles",
    "UAVs & Drones",
    "Legged Robots",
    "Imitation Learning for Robotics",

    # =========================================================================
    # VII. AI for Science
    # =========================================================================
    # Life Sciences
    "AI for Drug Discovery",
    "Protein Folding & Structure Prediction",
    "Molecular Generation & Docking",
    "Genomics & Bioinformatics",
    "Computational Pathology",
    "AI for Neuroscience",
    # Physics & Chem
    "Physics-Informed Machine Learning (PINNs)",
    "AI for Material Science / Material Discovery",
    "Computational Fluid Dynamics (CFD) with AI",
    "Quantum Machine Learning (QML)",
    "AI for High Energy Physics",
    "AI for Chemistry / Retrosynthesis",
    # Earth & Space
    "AI for Weather Forecasting & Meteorology",
    "Climate Modeling & Change Prediction",
    "AI for Astronomy & Astrophysics",
    "Seismic Analysis & Geophysics",

    # =========================================================================
    # VIII. Vertical & Social Applications
    # =========================================================================
    # Urban & Environment
    "AI for Urban Planning (Urban Computing)",
    "Smart Traffic & Transportation Systems",
    "AI for Smart Grid & Energy Optimization",
    "Sustainability & Green AI",
    "AI for Agriculture (Precision Farming)",
    # Healthcare (Clinical)
    "Clinical Decision Support Systems",
    "Electronic Health Records (EHR) Analysis",
    "Personalized Medicine",
    # Finance & Business
    "AI for Finance (FinTech)",
    "Algorithmic Trading & Stock Prediction",
    "Credit Scoring & Risk Assessment",
    "Fraud Detection",
    "Recommender Systems",
    # Other Verticals
    "AI for Education (EdTech)",
    "Intelligent Tutoring Systems",
    "AI for Law (LegalTech)",
    "AI for Cybersecurity",
    "AI for Art & Creativity",
    "AI for Social Good",

    # =========================================================================
    # IX. Trustworthy, Safety & Security
    # =========================================================================
    "AI Safety & Alignment",
    "Explainable AI (XAI) & Interpretability",
    "Adversarial Robustness (Attacks & Defenses)",
    "Fairness, Accountability, and Transparency (FAT)",
    "Privacy-Preserving Machine Learning",
    "Differential Privacy",
    "Federated Unlearning / Machine Unlearning",
    "Deepfake Detection & Synthesis",
    "AI Watermarking & Copyright Protection",
    "Uncertainty Quantification",
    "Out-of-Distribution (OOD) Detection",
    "Anomaly Detection",
    "Hallucination Mitigation",

    # =========================================================================
    # X. Systems, Hardware & Engineering
    # =========================================================================
    "MLOps (Machine Learning Operations)",
    "LLMOps",
    "Edge AI / TinyML",
    "On-Device Learning",
    "Model Compression (Pruning, Quantization, Distillation)",
    "Efficient Inference",
    "Distributed Training & Systems",
    "Hardware-Aware Neural Architecture Search (HW-NAS)",
    "AI Accelerators & Chip Design",
    "Neuromorphic Computing",

    # =========================================================================
    # XI. Geometric DL & Topology
    # =========================================================================
    "Geometric Deep Learning",
    "Equivariant Neural Networks",
    "Topological Data Analysis (TDA)",
    "Persistent Homology",
    "Hyperbolic Neural Networks",
    "Riemannian Optimization",
    "Group Equivariant CNNs",
    "Symmetry-Preserving Machine Learning",

    # =========================================================================
    # XII. Data-Centric AI & Data Engineering
    # =========================================================================
    "Data-Centric AI",
    "Synthetic Data Generation",
    "Data Pruning & Coreset Selection",
    "Dataset Distillation / Condensation",
    "Data Valuation (e.g., Shapley Values)",
    "Label Noise Learning",
    "Automated Data Cleaning & Curation",
    "Data Augmentation Strategies",
    "Weak Supervision & Snorkel",

    # =========================================================================
    # XIII. Affective Computing & HRI
    # =========================================================================
    "Affective Computing (Emotion AI)",
    "Sentiment & Emotion Analysis (Multimodal)",
    "Micro-expression Recognition",
    "Brain-Computer Interfaces (BCI) Signal Processing",
    "Gaze Estimation & Eye Tracking",
    "Gesture Recognition",
    "Haptic Perception & Rendering (Touch AI)",
    "Social Robotics",
    "Machine Theory of Mind",
    "Persuasive Technology",

    # =========================================================================
    # XIV. Agents, Game Theory & Complex Systems
    # =========================================================================
    "Multi-Agent Path Finding (MAPF)",
    "Cooperative AI / Cooperative MARL",
    "Mechanism Design & Algorithmic Game Theory",
    "Computational Social Choice",
    "Agent-Based Modeling (ABM)",
    "Tool Learning / Tool-Augmented LLMs",
    "Autonomous Web Agents",
    "Procedural Content Generation (PCG) in Games",
    "Game AI (Non-RL approaches)",

    # =========================================================================
    # XV. Niche Vertical Applications
    # =========================================================================
    # --- Humanities & Heritage ---
    "Digital Humanities",
    "AI for Archaeology & Cultural Heritage Restoration",
    "Computational History",
    "AI for Linguistics (Computational Linguistics)",
    # --- Industrial & Manufacturing ---
    "Predictive Maintenance",
    "Digital Twins",
    "Industrial Internet of Things (IIoT) AI",
    "Supply Chain Optimization",
    "Operations Research with ML",
    # --- Life Style & Entertainment ---
    "AI for Fashion & Virtual Try-On",
    "AI for Sports Analytics",
    "Computational Photography",
    "AI for Journalism & Media",
    # --- Earth & Bio Niches ---
    "Bioacoustics (AI for Animal Communication)",
    "Computational Ecology",
    "Oceanography & Marine AI",
    "Wildfire Prediction & Management",

    # =========================================================================
    # XVI. Neuromorphic & Unconventional Computing
    # =========================================================================
    "Neuromorphic Computing Algorithms",
    "Reservoir Computing / Echo State Networks",
    "Hyperdimensional Computing (HDC)",
    "Optical Neural Networks",
    "Biological Learning Rules (Hebbian, STDP)",
    "Energy-Based Models (EBMs)",
    "Quantum Control with ML",

    # =========================================================================
    # XVII. World Models & Predictive Architectures
    # =========================================================================
    "World Models",
    "Joint Embedding Predictive Architectures (JEPA / I-JEPA / V-JEPA)",
    "Model-Based Reinforcement Learning (MBRL)",
    "Video Prediction & Future Frame Prediction",
    "Physics-based Simulation & Neural Physics",
    "Predictive Coding (Neuroscience-inspired)",
    "Latent Dynamics Models",
    "Object-Centric Learning / Object-Centric Representations",
    "Causal Discovery & Causal Reasoning",

    # =========================================================================
    # XVIII. Advanced Reasoning, Planning & System 2
    # =========================================================================
    "Reasoning & Inference",
    "Chain of Thought (CoT) & Tree of Thoughts (ToT)",
    "Search & Planning Algorithms (MCTS, A*) in LLMs",
    "Logical Reasoning & Theorem Proving",
    "Mathematical Reasoning",
    "Program Synthesis & Neural Program Induction",
    "Algorithmic Reasoning",

    # =========================================================================
    # XIX. Memory, Retrieval & Long-Context
    # =========================================================================
    "Long-Context LLMs / Infinite Context",
    "External Memory Systems / Neural Turing Machines",
    "Vector Databases & Neural Retrieval",
    "Episodic & Semantic Memory",
    "Knowledge Editing & Model Editing",

    # =========================================================================
    # XX. Embodied Simulation & Environment Interaction
    # =========================================================================
    "Embodied Agents in Simulation (e.g., Minecraft, Habitat)",
    "Procedural Environment Generation",
    "Open-Ended Learning / Open-Endedness",
    "Developmental Robotics / Developmental AI",
    "Affordance Learning"
]