import os
import sys
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def build_rebuttal():
    doc = docx.Document(r'C:/Users/felix/Downloads/IAIC Rebuttal Letter Template.docx')
    
    # We will clear existing paragraphs and rewrite following the exact IAIC template style and structure
    # Check styles
    print("Normal font:", doc.styles['Normal'].font.name)
    
    # Empty doc
    for p in list(doc.paragraphs):
        p_elem = p._p
        p_elem.getparent().remove(p_elem)
        
    def add_meta(label, val):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        r1 = p.add_run(label + ": ")
        r1.bold = True
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(11)
        r2 = p.add_run(val)
        r2.font.name = 'Times New Roman'
        r2.font.size = Pt(11)

    add_meta("Manuscript ID", "193")
    add_meta("Title", "History-Conditioned SAC under Sensor Faults: A Pilot Study of Adaptation and Retention")
    add_meta("Conference / Proceedings", "2nd International Conference on Digital Technology (ICoDiT 2026) / Springer CCIS")
    add_meta("Authors", "Felix Surjodinoto*, Keandre Rafael, Delroy, Samuel Philip")
    add_meta("Affiliation", "Computer Science Department, BINUS Online Learning, Bina Nusantara University, Jakarta, Indonesia")
    add_meta("Corresponding Author Email", "felix.surjodinoto@binus.ac.id")

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(6)

    p_dear = doc.add_paragraph()
    p_dear.paragraph_format.space_after = Pt(6)
    r = p_dear.add_run("Dear ICoDiT 2026 Program Chairs, Track Editors, and Reviewers,")
    r.bold = True
    r.font.name = 'Times New Roman'
    r.font.size = Pt(11)

    p_intro = doc.add_paragraph()
    p_intro.paragraph_format.space_after = Pt(8)
    p_intro.paragraph_format.line_spacing = 1.15
    r = p_intro.add_run(
        "We sincerely thank the Program Committee and the reviewers for their positive evaluation, thorough reading, "
        "and exceptionally constructive feedback on our manuscript. We have carefully considered all comments, suggestions, "
        "and requested clarifications, and have comprehensively revised the manuscript. "
        "Below is our point-by-point response detailing every modification made. "
        "Reviewer comments are in bold, and our responses follow in regular text. "
        "In accordance with conference instructions, all modifications are explicitly highlighted in the accompanying "
        "Track Changes manuscript (File 2: 193 - Revised Article (Track Changes).pdf), and the finalized clean manuscript is provided as File 3."
    )
    r.font.name = 'Times New Roman'
    r.font.size = Pt(11)

    def add_section_header(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(title)
        r.bold = True
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    def add_comment_response(c_num, comment_text, response_paragraphs):
        p_c = doc.add_paragraph()
        p_c.paragraph_format.space_before = Pt(6)
        p_c.paragraph_format.space_after = Pt(3)
        p_c.paragraph_format.line_spacing = 1.15
        r_lbl = p_c.add_run(f"Comment {c_num}: ")
        r_lbl.bold = True
        r_lbl.font.name = 'Times New Roman'
        r_lbl.font.size = Pt(11)
        r_txt = p_c.add_run(comment_text)
        r_txt.bold = True
        r_txt.italic = True
        r_txt.font.name = 'Times New Roman'
        r_txt.font.size = Pt(11)

        for idx, resp in enumerate(response_paragraphs):
            p_r = doc.add_paragraph()
            p_r.paragraph_format.space_before = Pt(0)
            p_r.paragraph_format.space_after = Pt(4)
            p_r.paragraph_format.line_spacing = 1.15
            if idx == 0:
                r_lbl2 = p_r.add_run("Response: ")
                r_lbl2.bold = True
                r_lbl2.font.name = 'Times New Roman'
                r_lbl2.font.size = Pt(11)
            r_resp = p_r.add_run(resp)
            r_resp.font.name = 'Times New Roman'
            r_resp.font.size = Pt(11)

    # ------------------ REVIEWER 4 ------------------
    add_section_header("Reviewer 4")

    add_comment_response(
        1,
        "Benchmark diversity and environment breadth: Can the authors evaluate on additional MuJoCo tasks such as Humanoid-v5 or Ant-v5 to assess generalizability across locomotion dynamics?",
        [
            "We thank the reviewer for this crucial suggestion regarding benchmark diversity. In response, we have implemented and incorporated the Ant-v5 benchmark into the complete study protocol with identical paired-seed experimental rigour (Section 2.1, 3.3, 4; Tables 3-5; Figure 2).",
            "Ant-v5 features an eight-legged locomotion morphology with a 27-dimensional observation space (excluding contact forces). We evaluated eight source seeds per controller (reactive MLP vs. history-conditioned GPT-style causal transformer) across all 5 experimental branches (40 reactive + 40 transformer = 80 branches), each budgeted to 20,000 steps and 5,000 parameter updates.",
            "Under the fixed, un-tuned 400,000-step source training recipe, all Ant source controllers learned a stable standing posture (mean forward velocity within ±0.007 m/s). As a result, sensor noise and observation delays inflicted negligible return penalties on the standing posture. Rather than retroactively hyperparameter-tuning the recipe to fit our narrative, we transparently report this empirical finding in Section 3.3 as an instructive case of task morphology sensitivity.",
            "Regarding Humanoid-v5, we explicitly acknowledge in Section 4 (Limitations) that Humanoid was not evaluated because standard SAC benchmark baselines require several millions of interaction steps—far exceeding our standardized 400,000-step source budget—and due to the severe VRAM/compute constraints of our physical workstation during the synchronous 160-branch execution."
        ]
    )

    add_comment_response(
        2,
        "Off-policy bias and soft Bellman target distortion: How does mixing clean source replay transitions with corrupted online transitions affect critic learning and value overestimation?",
        [
            "We thank the reviewer for this profound theoretical insight into off-policy reinforcement learning under distribution shift. We have developed and added an entirely new critic diagnostic framework in Section 2.5 and Section 3.4 (with empirical results tabulated in Table 5).",
            "Specifically, for every branch, both soft Q-critics were evaluated on held-out clean and corrupted trajectories at the initial source checkpoint and after 20,000 steps of continued training. We evaluated the held-out soft Bellman residual (using the saved temperature parameter alpha) and formulated an exploratory Monte-Carlo value-bias gap (bias on corrupted data minus bias on clean data), which isolates fault-specific overestimation from soft-value offsets.",
            "Key Empirical Finding: In HalfCheetah under observation delay, source critics severely overvalued delayed history states relative to clean states (+171 and +268 return units). Training on corrupted transitions alone shrank this gap to +27 and +114. However, maintaining 50% clean replay buffer mixing kept the gap elevated at +89 and +204 (paired effect: +62 [16, 109] and +90 [41, 139], with 7 of 8 paired seeds exhibiting positive gaps). Furthermore, replay mixing significantly increased the reactive critic's residual on corrupted transitions across all 8 seeds.",
            "This provides concrete mechanistic evidence of soft Bellman target distortion: mixing stale clean transitions impedes critic adaptation under non-Markovian observation delay. Under Gaussian noise and in the Ant environment, by contrast, the gap remained negligible."
        ]
    )

    add_comment_response(
        3,
        "Inference latency and real-time viability: Is history-conditioned transformer inference practical for high-rate continuous robotic control loops (100–500 Hz)?",
        [
            "We appreciate the reviewer raising this vital engineering consideration. In Section 2.6 and Section 3.5 (Table 6), we present benchmarked wall-clock inference latencies measured across 10,000 real model forward passes at deployment batch size 1 (excluding physics simulator overhead).",
            "On a single CPU thread (Intel i7-8700), our history-conditioned causal transformer achieves a mean latency of 0.58 ms (99th percentile: 0.66 ms, maximum: 1.56 ms). Because the maximum observed latency is strictly below 2.0 ms, the transformer comfortably satisfies the hard timing deadline for 500 Hz real-time robotic control. On 4 CPU threads, latency is 0.64 ms. The reactive MLP achieves 0.09 ms.",
            "Interestingly, GPU execution at batch size 1 incurs kernel-launch overhead (mean 1.64 ms), demonstrating that single-threaded CPU execution is optimal for edge deployment of low-parameter sequence models."
        ]
    )

    add_comment_response(
        4,
        "Reproducibility and artifact availability: Please provide complete code, seed configurations, and checkpoints to verify replication.",
        [
            "We have made the complete codebase, environment wrappers, seed generation scripts, launch scripts, frozen evaluation protocols, raw step-by-step evaluation records, and both source and adapted model checkpoints publicly available in our open repository at: https://github.com/Fane-Nathan/tmrl.",
            "Exactness verification: All 24 HalfCheetah pilot branches rerun with their original seeds reproduce their exact returns bit-for-bit (maximum difference 0.000). Nine branches rerun after workstation thermal interruptions produced bitwise-identical logs and checkpoints. Every branch executed exactly 20,000 steps and 5,000 gradient updates."
        ]
    )

    # ------------------ REVIEWER 5 ------------------
    add_section_header("Reviewer 5")

    add_comment_response(
        1,
        "Architecture and configuration specifications: The manuscript should provide full architectural details of the transformer, history representation, and fault injection mechanisms.",
        [
            "We have expanded Section 2.1–2.3 and added two comprehensive specification tables (Tables 1 and 2):",
            "1. Architectural Specifications (Table 1): Full component breakdown, embedding dimensionality (d_model = 64), 2 causal self-attention blocks (4 heads, causal masking), LayerNorm, MLP feature projection head (64 units), and parameter counts (87,624 actor / 92,290 critic parameters for transformer; 71,942 / 72,450 for reactive MLP). Episode boundary resets and sliding context buffering (K = 10 steps) are described in Section 2.2.",
            "2. Fault Injection Logic (Section 2.1): Precise mathematical models for Gaussian noise (zero-mean, covariance sigma^2 I with sigma = 0.15) and observation delay (FIFO ring buffer of depth d = 3 with state repeating during warm-up).",
            "3. Training Configurations (Table 2): Complete hyperparameter specification including learning rate (3e-4), optimizer (Adam), discount factor (gamma = 0.99), target smoothing (tau = 0.005), batch size (256), and replay proportions."
        ]
    )

    add_comment_response(
        2,
        "Statistical power and sample size: The evaluation budget and seed counts should be increased to ensure statistically sound conclusions.",
        [
            "We have significantly scaled both sample size and evaluation horizons:",
            "1. Sample Size: Increased from 3 seeds to 8 independent source seeds per controller and task (Section 2.3). With 2 tasks, 2 architectures, and 5 branches, this represents 160 fully trained and logged policy branches.",
            "2. Training Horizon: Extended by 500% from 4,000 steps to 20,000 environment steps (5,000 updates), evaluated across 7 discrete checkpoints (0, 2k, 4k, 6k, 10k, 15k, 20k steps; Figure 2).",
            "3. Formal Significance Testing (Section 2.4; Tables 3-4): Pre-specified exact two-sided Wilcoxon signed-rank tests paired on source seed, with family-wise error rate controlled via the Holm-Bonferroni step-down procedure, accompanied by Cohen's d_z effect sizes and both parametric 95% t-intervals and non-parametric BCa bootstrap confidence intervals."
        ]
    )

    add_comment_response(
        3,
        "Comparative analysis between Transformer and Reactive MLP: Why does the transformer exhibit different adaptation characteristics under delay versus noise?",
        [
            "In Section 4 (Discussion), we have added a dedicated analytical subsection dissecting controller observability and behavioral trade-offs.",
            "Under Gaussian noise, averaging across the temporal history window acts as a natural low-pass filter, enabling the transformer to match reactive MLP adaptation while mitigating policy jitter. Under unannounced observation delay, however, the temporal context contains valid past states that no longer correspond to the current physical plant state.",
            "Because standard self-attention calculates uniform attention over all past positions, off-policy critic gradient updates suffer target distortion unless the model actively learns to downweight recent tokens. We formulate four testable hypotheses and outline future ablation directions (positional encoding decay, explicit delay detection heads) to further isolate these dynamics."
        ]
    )

    # ------------------ ADDITIONAL EDITORIAL REQUIREMENTS ------------------
    add_section_header("Additional Editorial and Publisher Requirements")

    add_comment_response(
        1,
        "Verification of references, declaration of AI assistance, author contributions, and layout constraints.",
        [
            "1. Reference Audit: We performed an exhaustive audit of all 18 cited works. All citations have been cross-checked against primary publication records (NeurIPS, ICML, ICLR, CoRL, JMLR). No retracted literature or unverified preprints are included.",
            "2. AI Usage Declaration: An explicit declaration is provided in Section 4 (page 14) specifying the roles of Google Antigravity (Gemini), OpenAI Codex, and Claude Code (Anthropic) in code implementation, test orchestration, and manuscript structuring. All experimental results, statistical tests, and data logs are strictly simulator-generated.",
            "3. Author Contributions (CRediT): In accordance with editorial instructions, an Author Contributions section has been incorporated directly following the AI Usage Declaration (page 14), delineating roles in conceptualization, software development, formal analysis, and supervision.",
            "4. Page Budget & Header Formatting: The revised manuscript strictly adheres to the Springer LNCS page limit (exactly 15 pages). Even-page running headers have been cleaned to suppress stray author header strings while preserving correct running pagination."
        ]
    )

    p_close = doc.add_paragraph()
    p_close.paragraph_format.space_before = Pt(12)
    p_close.paragraph_format.space_after = Pt(8)
    p_close.paragraph_format.line_spacing = 1.15
    r = p_close.add_run(
        "We believe these substantial experimental additions, theoretical critic diagnostics, and detailed revisions have "
        "thoroughly addressed all reviewer recommendations and significantly strengthened the manuscript. "
        "We thank the reviewers and editors once again for their time and invaluable guidance."
    )
    r.font.name = 'Times New Roman'
    r.font.size = Pt(11)

    p_sign = doc.add_paragraph()
    p_sign.paragraph_format.space_before = Pt(12)
    p_sign.paragraph_format.space_after = Pt(2)
    r = p_sign.add_run("Sincerely,")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(11)

    p_auth = doc.add_paragraph()
    p_auth.paragraph_format.space_after = Pt(1)
    r = p_auth.add_run("Felix Surjodinoto (Corresponding Author)\nOn behalf of all co-authors: Keandre Rafael, Delroy, Samuel Philip\nBina Nusantara University, Jakarta, Indonesia\nEmail: felix.surjodinoto@binus.ac.id\nNovember 4, 2026")
    r.font.name = 'Times New Roman'
    r.font.size = Pt(11)

    out_path = r'paper/reporting/193 - Rebuttal Letter.docx'
    doc.save(out_path)
    print("Saved:", out_path, "Size:", os.path.getsize(out_path))

if __name__ == '__main__':
    build_rebuttal()
