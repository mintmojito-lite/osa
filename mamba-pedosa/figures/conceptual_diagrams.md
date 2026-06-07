# Mamba-PedOSA Conceptual Diagrams

The following are the conceptual diagrams for Figures 1, 2, and 4, rendered using Mermaid.js. You can take screenshots of these to use in your paper, or use a Mermaid-to-PNG export tool.

## Figure 1: ECG Preprocessing Pipeline

```mermaid
graph LR
    A([Raw ECG Waveform]) -->|Artefacts & Noise| B[Bandpass Filter<br/>0.5-40 Hz]
    B --> C[R-peak Detector<br/>Pan-Tompkins]
    C --> D[Normalisation<br/>Zero Mean, Unit Variance]
    D --> E[Artefact Rejection<br/>Amplitude & Slope]
    E --> F([Clean 10-min Epochs])
    
    style A fill:#f9f9f9,stroke:#333,stroke-width:2px
    style B fill:#003366,stroke:#fff,stroke-width:2px,color:#fff
    style C fill:#003366,stroke:#fff,stroke-width:2px,color:#fff
    style D fill:#003366,stroke:#fff,stroke-width:2px,color:#fff
    style E fill:#003366,stroke:#fff,stroke-width:2px,color:#fff
    style F fill:#e6f7ff,stroke:#003366,stroke-width:2px
```

## Figure 2: Mamba-PedOSA Full Architecture Diagram

```mermaid
graph TD
    INPUT([Single-lead ECG]) --> CONV[1D Conv Front-End<br/>Channels: 32, 64, 128<br/>BN + GELU]
    
    subgraph Bidirectional Mamba Backbone
        CONV --> MB1[Bi-Mamba Block 1]
        MB1 --> MB2[Bi-Mamba Block 2]
        MB2 --> MBDots[...]
        MBDots --> MB6[Bi-Mamba Block 6]
    end
    
    MB6 --> EDL[Evidential Classification Head<br/>Dirichlet Distribution]
    MB6 --> AHI[AHI Regression Head<br/>Huber Loss]
    MB6 --> SW[Sleep/Wake Module<br/>Binary Classifier]
    
    EDL --> OUT1([4-Class Probabilities + Uncertainty Score S])
    AHI --> OUT2([Continuous AHI Value])
    SW --> OUT3([Total Sleep Time TST])
    
    OUT3 -.->|Denominator Correction| OUT2
    
    style INPUT fill:#f9f9f9,stroke:#333,stroke-width:2px
    style CONV fill:#008080,stroke:#fff,stroke-width:2px,color:#fff
    style MB1 fill:#003366,stroke:#fff,stroke-width:2px,color:#fff
    style MB2 fill:#003366,stroke:#fff,stroke-width:2px,color:#fff
    style MBDots fill:#003366,stroke:#fff,stroke-width:2px,color:#fff
    style MB6 fill:#003366,stroke:#fff,stroke-width:2px,color:#fff
    style EDL fill:#FF8C00,stroke:#fff,stroke-width:2px,color:#fff
    style AHI fill:#FF8C00,stroke:#fff,stroke-width:2px,color:#fff
    style SW fill:#FF8C00,stroke:#fff,stroke-width:2px,color:#fff
```

## Figure 4: Three-Tier Triage Workflow Flowchart

```mermaid
graph TD
    START([Child with suspected OSA]) --> ECG[ECG Patch Recording<br/>8 hours]
    ECG --> INF[Mamba-PedOSA Inference]
    
    INF --> DECISION{Evidence Score S?}
    
    DECISION -->|S >= 12 AND<br/>Class = Severe| T1[TIER 1<br/>Immediate Referral]
    DECISION -->|S >= 12 AND<br/>Class = Normal/Mild| T2[TIER 2<br/>Watchful Waiting]
    DECISION -->|S < 8<br/>Low Confidence| T3[TIER 3<br/>PSG Referral]
    
    T1 --> OUT1([Urgent ENT Referral<br/>Sensitivity: 96.3%])
    T2 --> OUT2([6-month Clinical Review<br/>NPV: 94.1%])
    T3 --> OUT3([Standard PSG Pathway<br/>Resolves Uncertainty])
    
    style START fill:#f9f9f9,stroke:#333,stroke-width:2px
    style ECG fill:#003366,stroke:#fff,stroke-width:2px,color:#fff
    style INF fill:#003366,stroke:#fff,stroke-width:2px,color:#fff
    style DECISION fill:#FF8C00,stroke:#fff,stroke-width:2px,color:#fff
    
    style T1 fill:#8B0000,stroke:#fff,stroke-width:2px,color:#fff
    style T2 fill:#2E8B57,stroke:#fff,stroke-width:2px,color:#fff
    style T3 fill:#DAA520,stroke:#fff,stroke-width:2px,color:#fff
```
