import os
import sys
import copy
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import win32com.client

def build_presentation():
    template_path = os.path.abspath('paper/template.pptx')
    out_pptx = os.path.abspath('paper/reporting/193 - Presentation.pptx')
    out_pdf = os.path.abspath('paper/reporting/193 - Presentation.pdf')
    
    prs = Presentation(template_path)
    print(f"Loaded template. Initial slide count: {len(prs.slides)}")
    print(f"Slide width: {prs.slide_width}, height: {prs.slide_height}")
    
    # ------------------ SLIDE 1: TITLE SLIDE ------------------
    s1 = prs.slides[0]
    for shp in s1.shapes:
        if shp.has_text_frame:
            txt = shp.text.strip()
            if "Design Refinement" in txt:
                shp.text = ""
                p = shp.text_frame.paragraphs[0]
                p.text = "History-Conditioned SAC under Sensor Faults:\nA Pilot Study of Adaptation and Retention"
                p.font.name = "Montserrat"
                p.font.size = Pt(28)
                p.font.bold = True
                p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shp.top = Pt(150)
                shp.height = Pt(170)
            elif "Name" in txt:
                shp.text = ""
                p = shp.text_frame.paragraphs[0]
                p.text = "Felix Surjodinoto*, Keandre Rafael, Delroy, Samuel Philip"
                p.font.name = "Montserrat"
                p.font.size = Pt(20)
                p.font.bold = True
                p.font.color.rgb = RGBColor(0x01, 0x13, 0x4D)
                shp.top = Pt(455)
                shp.height = Pt(45)
            elif "Affiliation" in txt:
                shp.text = ""
                p = shp.text_frame.paragraphs[0]
                p.text = "Computer Science Department, BINUS Online Learning\nBina Nusantara University, Jakarta, Indonesia\nICoDiT 2026  |  Paper ID: 193  |  Springer Nature CCIS"
                p.font.name = "Open Sans"
                p.font.size = Pt(16)
                p.font.bold = False
                p.font.color.rgb = RGBColor(0x01, 0x13, 0x4D)
                shp.top = Pt(510)
                shp.height = Pt(90)

    # Base Slide 2 template
    s2_template = prs.slides[1]

    # Content slides definition
    content_slides = [
        {
            "title": "Motivation & Research Problem",
            "subtitle": "The Fragility of Deep Reinforcement Learning Under Unannounced Sensor Degradation",
            "bullets": [
                ("Physical Deployment Gap", "Deep RL policies trained in clean simulated environments suffer catastrophic failure when deployed on physical robots subject to hardware aging, calibration drift, and sensory latency."),
                ("Sensor Faults in the Wild", "Real-world robotic control regularly encounters zero-mean sensor noise (vibrations, thermal noise) and observation delays (transport latency, multi-rate sensor polling)."),
                ("The Adaptation vs. Retention Dilemma", "While policies can be fine-tuned online to restore degraded return, adaptation often triggers catastrophic forgetting of nominal clean behavior."),
                ("Research Core Question", "Can history-conditioned sequence models (causal transformers) adapt more robustly than reactive MLPs? Does replay buffer mixing successfully retain nominal performance without causing off-policy bias?")
            ]
        },
        {
            "title": "Problem Formulation & Fault Modeling",
            "subtitle": "Formalizing Unannounced Sensor Degradation in Continuous Locomotion (POMDP)",
            "bullets": [
                ("Partially Observable Formulation (POMDP)", "Continuous control modeled as M = (S, A, T, R, Omega, O, gamma), where the policy observes degraded observations rather than full ground-truth states."),
                ("Fault Model 1: Gaussian Sensor Noise", "Observation perturbed by zero-mean Gaussian noise: s_tilde_t = s_t + e_t, e_t ~ N(0, sigma^2 I) with sigma = 0.15. High-frequency perturbation without temporal lag."),
                ("Fault Model 2: Observation Delay", "FIFO ring buffer of depth d = 3 (s_tilde_t = s_t-d). Introduces non-Markovian state transition dynamics and breaks immediate velocity tracking."),
                ("Dual Evaluation Protocol", "1. Adaptation Return: Policy performance evaluated under active continuous sensor fault.\n2. Retention Return: Retained policy competence on the original clean environment without re-training.")
            ]
        },
        {
            "title": "Architectures & Replay Buffer Strategy",
            "subtitle": "History-Conditioned Transformer vs. Reactive MLP with Nominal Replay Mixing",
            "bullets": [
                ("History-Conditioned Controller (Transformer)", "GPT-style causal transformer; sliding context window K = 10 steps; token embedding d_model = 64; 2 causal self-attention blocks with 4 heads; MLP projection head. 87.6k actor / 92.3k critic parameters."),
                ("Reactive Controller Baseline (MLP)", "Standard SAC policy: 2 hidden layers with 256 units each, ReLU activations. 71.9k actor / 72.5k critic parameters. Memoryless mapping s_t -> a_t."),
                ("Replay Buffer Strategy (50% Mixing)", "To mitigate catastrophic forgetting, 50% of the training minibatch is sampled from offline clean nominal transitions and 50% from online corrupted transitions."),
                ("Hypothesis Tested", "Does memory retention come at the expense of adaptation velocity, and does mixing clean and corrupted transitions distort the soft Bellman target?")
            ]
        },
        {
            "title": "Rigorous Experimental Protocol",
            "subtitle": "160 Policy Branches Evaluated Across HalfCheetah-v5 and Ant-v5 Benchmarks",
            "image": os.path.abspath("paper/trackmania_pilot_workflow.png"),
            "bullets": [
                ("Phase 1: Source Policy Training", "8 independent source seeds per controller and task trained for 400,000 steps under nominal clean conditions (32 source baselines)."),
                ("Phase 2: Continued Training (20k Steps)", "5 parallel branches per seed: Clean Control, Noise Only, Noise + Replay, Delay Only, Delay + Replay (80 HalfCheetah + 80 Ant = 160 branches)."),
                ("Pre-Specified Significance Testing", "Exact Wilcoxon signed-rank tests paired on source seed; family-wise error rate controlled via Holm-Bonferroni correction (adjusted alpha floors: 0.031 and 0.0625); Cohen's d_z effect sizes and 95% bootstrap intervals.")
            ]
        },
        {
            "title": "HalfCheetah-v5: Adaptation & Retention",
            "subtitle": "Divergent Dynamics Between High-Frequency Noise and Temporal Observation Delay",
            "image": os.path.abspath("paper/fig_curves_hc.png"),
            "bullets": [
                ("Sensor Noise Resilience", "Both reactive MLP and transformer recover rapid near-optimal velocity. Replay mixing successfully mitigates forgetting, preserving nominal clean returns with minimal penalty."),
                ("Severe Degradation Under Delay", "Observation delay inflicts massive immediate performance collapse (>2,000 return penalty). Continued training achieves only partial recovery across 20,000 steps."),
                ("Transformer Attention Dilemma", "Under unannounced delay, the transformer exhibits greater performance vulnerability than MLP, attending to stale historical tokens that misrepresent current plant physics.")
            ]
        },
        {
            "title": "Ant-v5: Morphology & Task Sensitivity",
            "subtitle": "Morphological Postural Convergence and Re-Optimization Transients",
            "image": os.path.abspath("paper/fig_curves_ant.png"),
            "bullets": [
                ("Morphological Standing Posture", "Under the standardized 400,000-step source recipe, all 8 Ant seeds converged to an ultra-stable standing posture (forward velocity within +/-0.007 m/s)."),
                ("Fault Sensitivity Finding", "Because standing requires negligible dynamic velocity feedback, sensor noise and delay inflicted negligible performance penalties. Reported transparently rather than retroactively tuning hyperparameters."),
                ("Warm-Start Re-Optimization Dip", "All Ant branches (including clean controls) experienced an initial transient return drop in the first 2,000 steps, proving that optimizer re-initialization rather than sensor degradation caused the dip.")
            ]
        },
        {
            "title": "Mechanistic Critic Diagnostic",
            "subtitle": "Soft Bellman Target Distortion Explains Replay Failure Under Observation Delay",
            "bullets": [
                ("Held-Out Critic Evaluation Framework", "Both soft Q-critics evaluated on held-out clean and corrupted trajectories at source checkpoint and after 20,000 steps. Evaluated soft Bellman residual and Monte-Carlo value-bias gap (Bias_corrupt - Bias_clean)."),
                ("Massive Source Value Overestimation", "Source critics severely overvalued delayed history states (+171 and +268 return units in HalfCheetah)."),
                ("Distortion From Replay Mixing", "Training on corrupted data alone shrank the bias gap to +27 and +114. Mixing 50% clean replay maintained the gap elevated at +89 and +204 (paired effect: +62 [16, 109] and +90 [41, 139], 7/8 seeds positive)."),
                ("Core Theoretical Insight", "Mixing stale clean transitions creates an off-policy distribution mismatch that actively distorts the soft Bellman target, proving why replay harmed adaptation under delay.")
            ]
        },
        {
            "title": "Real-Time Edge Viability: 500 Hz Benchmark",
            "subtitle": "Sub-Millisecond Inference Proves Feasibility for High-Speed Robotic Control",
            "table_data": [
                ["Platform", "Controller", "Threads", "Mean Latency", "99th %ile", "Max Latency", "500 Hz Viable?"],
                ["CPU (i7-8700)", "Transformer Actor", "1 thread", "0.58 ms", "0.66 ms", "1.56 ms", "YES (Budget < 2.0 ms)"],
                ["CPU (i7-8700)", "Reactive MLP Actor", "1 thread", "0.09 ms", "0.12 ms", "0.28 ms", "YES"],
                ["CPU (i7-8700)", "Transformer Actor", "4 threads", "0.64 ms", "0.78 ms", "1.82 ms", "YES"],
                ["GPU (RTX 3070 Ti)", "Transformer Actor", "CUDA", "1.64 ms", "1.92 ms", "3.14 ms", "MARGINAL (Kernel overhead)"]
            ],
            "bullets": [
                ("10,000 Timed Forward Passes", "Measured at deployment batch size 1 with real model inputs (excluding physics engine simulation overhead)."),
                ("Strict Real-Time Compliance", "Transformer inference on single-threaded CPU consistently completes in under 1.56 ms, comfortably within the 2.0 ms cycle budget of 500 Hz robotics."),
                ("Edge Architecture Takeaway", "Single-threaded CPU execution outperforms GPU at batch size 1, confirming suitability for embedded robot microcontrollers without dedicated discrete accelerators.")
            ]
        },
        {
            "title": "Conclusions & Key Insights",
            "subtitle": "Actionable Takeaways for Fault-Tolerant Reinforcement Learning in Physical Systems",
            "bullets": [
                ("1. History Conditioning is Fault-Specific", "Causal transformers act as effective temporal low-pass filters under high-frequency noise, but suffer severe attention confusion under observation delay unless delay is explicitly represented."),
                ("2. Replay Buffer Trade-Off", "50% nominal replay mixing successfully prevents catastrophic forgetting under noise, but induces severe soft Bellman target distortion and off-policy bias under non-Markovian delay."),
                ("3. Strict Real-Time Deployment", "Transformer history conditioning requires only 87k parameters and executes in 0.58 ms on a single CPU thread, satisfying real-time 500 Hz robotics."),
                ("4. Open Science & Provenance", "All code, environment wrappers, seed protocols, checkpoints, and evaluation records are publicly released at https://github.com/Fane-Nathan/tmrl.")
            ]
        },
        {
            "title": "Thank You & Discussion",
            "subtitle": "History-Conditioned SAC under Sensor Faults: A Pilot Study of Adaptation and Retention",
            "bullets": [
                ("Authors", "Felix Surjodinoto*, Keandre Rafael, Delroy, Samuel Philip"),
                ("Affiliation", "Computer Science Department, BINUS Online Learning, Bina Nusantara University, Jakarta, Indonesia"),
                ("Primary Contact", "felix.surjodinoto@binus.ac.id"),
                ("Artifact Repository", "https://github.com/Fane-Nathan/tmrl (Code, Checkpoints, Evaluation Logs)"),
                ("Conference", "2nd International Conference on Digital Technology (ICoDiT 2026) | Paper ID: 193")
            ]
        }
    ]

    # Process Slide 2 (index 1 in prs.slides) and create Slides 3 to 11
    # We will populate Slide 2 with content_slides[0]
    # For slides 3..11, we create a new slide with s2_template.slide_layout and clone all shapes from s2_template
    slides_to_populate = []
    slides_to_populate.append((s2_template, content_slides[0]))

    for cdata in content_slides[1:]:
        new_slide = prs.slides.add_slide(s2_template.slide_layout)
        for s in s2_template.shapes:
            el = copy.deepcopy(s.element)
            new_slide.shapes._spTree.append(el)
        slides_to_populate.append((new_slide, cdata))

    print(f"Total slides prepared: {len(prs.slides)}")

    for idx, (slide, cdata) in enumerate(slides_to_populate):
        s_num = idx + 2
        print(f"Populating Slide {s_num}: {cdata['title']}")
        
        # Delete the original placeholder body shape "Google Shape;117;p2"
        # and update title & subtitle
        for shp in list(slide.shapes):
            if shp.has_text_frame:
                txt = shp.text.strip()
                if "Title" == txt:
                    shp.text = ""
                    p = shp.text_frame.paragraphs[0]
                    p.text = cdata["title"]
                    p.font.name = "Montserrat"
                    p.font.size = Pt(30)
                    p.font.bold = True
                    p.font.color.rgb = RGBColor(0x00, 0x1E, 0x61)
                    shp.left = Pt(78)
                    shp.top = Pt(135)
                    shp.width = Pt(1100)
                    shp.height = Pt(45)
                elif "Subtitle" == txt:
                    shp.text = ""
                    p = shp.text_frame.paragraphs[0]
                    p.text = cdata["subtitle"]
                    p.font.name = "Montserrat"
                    p.font.size = Pt(20)
                    p.font.bold = True
                    p.font.color.rgb = RGBColor(0x1F, 0x48, 0x81)
                    shp.left = Pt(78)
                    shp.top = Pt(180)
                    shp.width = Pt(1250)
                    shp.height = Pt(35)
                elif "Body text" in txt:
                    sp_elem = shp.element
                    sp_elem.getparent().remove(sp_elem)

        has_image = "image" in cdata
        has_table = "table_data" in cdata

        if has_image:
            # Bullets on left
            bx = slide.shapes.add_textbox(Pt(78), Pt(225), Pt(680), Pt(520))
            tf = bx.text_frame
            tf.word_wrap = True
            for b_idx, (head, desc) in enumerate(cdata.get("bullets", [])):
                p = tf.paragraphs[0] if b_idx == 0 else tf.add_paragraph()
                p.space_after = Pt(12)
                p.line_spacing = 1.15
                
                r1 = p.add_run()
                r1.text = f"•  {head}: "
                r1.font.name = "Open Sans"
                r1.font.size = Pt(15)
                r1.font.bold = True
                r1.font.color.rgb = RGBColor(0x00, 0x1E, 0x61)
                
                r2 = p.add_run()
                r2.text = desc
                r2.font.name = "Open Sans"
                r2.font.size = Pt(15)
                r2.font.bold = False
                r2.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)

            # Image on right
            img_path = cdata["image"]
            img_left = Pt(780)
            img_top = Pt(230)
            img_w = Pt(580)
            pic = slide.shapes.add_picture(img_path, img_left, img_top, width=img_w)
            # Center vertically within budget if too tall
            max_h = Pt(500)
            if pic.height > max_h:
                scale = max_h / pic.height
                pic.width = int(pic.width * scale)
                pic.height = max_h
            pic.top = Pt(235 + (500 - pic.height / Pt(1)) / 2)

        elif has_table:
            # Table on top
            t_rows = len(cdata["table_data"])
            t_cols = len(cdata["table_data"][0])
            tbl_shape = slide.shapes.add_table(t_rows, t_cols, Pt(78), Pt(225), Pt(1280), Pt(190))
            tbl = tbl_shape.table
            
            # Set col widths
            col_widths = [Pt(160), Pt(230), Pt(120), Pt(140), Pt(130), Pt(140), Pt(360)]
            for c_i, w in enumerate(col_widths):
                tbl.columns[c_i].width = w

            for r_i, row in enumerate(cdata["table_data"]):
                for c_i, val in enumerate(row):
                    cell = tbl.cell(r_i, c_i)
                    cell.text = ""
                    p = cell.text_frame.paragraphs[0]
                    p.text = val
                    p.font.name = "Open Sans"
                    p.font.size = Pt(13)
                    if r_i == 0:
                        p.font.bold = True
                        p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = RGBColor(0x00, 0x1E, 0x61)
                    else:
                        p.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
                        cell.fill.solid()
                        if r_i % 2 == 1:
                            cell.fill.fore_color.rgb = RGBColor(0xF2, 0xF5, 0xF9)
                        else:
                            cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

            # Bullets below table
            bx = slide.shapes.add_textbox(Pt(78), Pt(455), Pt(1280), Pt(280))
            tf = bx.text_frame
            tf.word_wrap = True
            for b_idx, (head, desc) in enumerate(cdata.get("bullets", [])):
                p = tf.paragraphs[0] if b_idx == 0 else tf.add_paragraph()
                p.space_after = Pt(10)
                p.line_spacing = 1.15
                
                r1 = p.add_run()
                r1.text = f"•  {head}: "
                r1.font.name = "Open Sans"
                r1.font.size = Pt(16)
                r1.font.bold = True
                r1.font.color.rgb = RGBColor(0x00, 0x1E, 0x61)
                
                r2 = p.add_run()
                r2.text = desc
                r2.font.name = "Open Sans"
                r2.font.size = Pt(16)
                r2.font.bold = False
                r2.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)

        else:
            # Full width bullets
            bx = slide.shapes.add_textbox(Pt(78), Pt(230), Pt(1280), Pt(520))
            tf = bx.text_frame
            tf.word_wrap = True
            for b_idx, (head, desc) in enumerate(cdata.get("bullets", [])):
                p = tf.paragraphs[0] if b_idx == 0 else tf.add_paragraph()
                p.space_after = Pt(14)
                p.line_spacing = 1.2
                
                r1 = p.add_run()
                r1.text = f"•  {head}: "
                r1.font.name = "Open Sans"
                r1.font.size = Pt(17)
                r1.font.bold = True
                r1.font.color.rgb = RGBColor(0x00, 0x1E, 0x61)
                
                r2 = p.add_run()
                r2.text = desc
                r2.font.name = "Open Sans"
                r2.font.size = Pt(17)
                r2.font.bold = False
                r2.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)

    prs.save(out_pptx)
    print(f"Saved PPTX successfully: {out_pptx} (Size: {os.path.getsize(out_pptx)} bytes)")

    # Export PDF via PowerPoint COM
    print("Exporting PDF via PowerPoint...")
    ppt = win32com.client.Dispatch("PowerPoint.Application")
    ppt.Visible = True
    pres = ppt.Presentations.Open(out_pptx)
    pres.SaveAs(out_pdf, 32) # 32 = ppSaveAsPDF
    pres.Close()
    ppt.Quit()
    print(f"Saved PDF successfully: {out_pdf} (Size: {os.path.getsize(out_pdf)} bytes)")

if __name__ == '__main__':
    build_presentation()
