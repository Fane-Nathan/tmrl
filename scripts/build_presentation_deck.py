import os
import sys
import copy
import shutil
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import win32com.client

def build_presentation():
    template_path = os.path.abspath('paper/template.pptx')
    out_pptx = os.path.abspath('paper/reporting/193 - Presentation.pptx')
    out_pdf = os.path.abspath('paper/reporting/193 - Presentation.pdf')
    onedrive_pdf = os.path.abspath('paper/reporting/ICoDiT 2026 - Paper 193 Presentation Slides.pdf')
    local_slides_pdf = os.path.abspath('paper/slides.pdf')
    
    prs = Presentation(template_path)
    print(f"Loaded template. Initial slide count: {len(prs.slides)}")
    
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
                p.text = "Computer Science Department, School of Computer Science\nBina Nusantara University, Jakarta, Indonesia\nICoDiT 2026  |  Paper ID: 193  |  Springer Nature CCIS"
                p.font.name = "Open Sans"
                p.font.size = Pt(16)
                p.font.bold = False
                p.font.color.rgb = RGBColor(0x01, 0x13, 0x4D)
                shp.top = Pt(510)
                shp.height = Pt(90)

    # Base Slide 2 template
    s2_template = prs.slides[1]

    # Content slides definition deeply focused on the research questions, why this research, and its purpose
    content_slides = [
        {
            "title": "Why This Research? The Origin & Motivation",
            "subtitle": "Training Plateaus in Real-Time TrackMania 2020 and the Mystery of Sensory Shifts",
            "image": os.path.abspath("paper/trackmania_pilot_workflow.png"),
            "bullets": [
                ("The High-Speed Challenge", "Our overarching goal is real-time visual control in TrackMania 2020 at 100–500 Hz. In this domain, sequence models and world models repeatedly encountered severe performance plateaus."),
                ("The Diagnostic Dilemma", "Why did policies stall? Did they suffer from loss of plasticity (inability to learn further)? Primacy bias (overfitting early data)? Or catastrophic forgetting during fine-tuning?"),
                ("The Physical Reality of Sensors", "Physical deployment introduces unannounced sensory degradation: camera vibration (noise) and transmission lag (delay). Actuator faults change control; sensor faults change information."),
                ("Need for Controlled Isolation", "In end-to-end visual driving, these factors are deeply entangled. We needed a rigorous, controlled MuJoCo benchmark to isolate and measure adaptation and retention directly.")
            ]
        },
        {
            "title": "Core Research Questions (What We Set Out to Answer)",
            "subtitle": "Decoupling Adaptation, Catastrophic Forgetting, and Replay Buffer Dynamics",
            "bullets": [
                ("RQ1: Adaptation vs. Retention Trade-off", "When a trained policy encounters unannounced sensory degradation, can continued training restore return (Adaptation), and does fine-tuning erase performance on nominal clean conditions (Retention)?"),
                ("RQ2: Architectural Inductive Bias", "Does history conditioning via a sequence model (Causal Transformer) provide an architectural buffer against sensor faults compared to a memoryless reactive MLP, or does past context misinform under latency?"),
                ("RQ3: The Replay Buffer Dilemma", "Does the standard continual learning remedy—mixing prior clean experience into replay—prevent forgetting, or does it introduce an off-policy distribution mismatch that corrupts Bellman value targets?"),
                ("RQ4: Real-Time Edge Feasibility", "Can an expressive history-conditioned transformer execute within the strict sub-2ms cycle budget required for 500 Hz physical robotic control on edge CPU hardware?")
            ]
        },
        {
            "title": "Experimental Formulation & Fault Modeling",
            "subtitle": "Partially Observable Formulation (POMDP) and Controlled 5-Branch Protocol",
            "bullets": [
                ("POMDP Formulation", "Continuous control modeled as M = (S, A, T, R, Omega, O, gamma), where the policy receives degraded observations s_tilde_t rather than ground-truth physical states."),
                ("Fault Model 1: Gaussian Noise", "s_tilde_t = s_t + e_t, e_t ~ N(0, sigma^2 I) with sigma = 0.15. Perturbs signal amplitude without introducing temporal lag."),
                ("Fault Model 2: Observation Delay", "FIFO ring buffer of depth d = 3 (s_tilde_t = s_t-d). Breaks Markovian velocity tracking; observation represents outdated physical plant state."),
                ("5-Branch Experimental Design", "1. Clean Control (isolates optimization drift)\n2. Noise Only (pure adaptation)\n3. Noise + Replay (rehearsal)\n4. Delay Only (pure adaptation)\n5. Delay + Replay (rehearsal)"),
                ("Paired-Seed Statistical Rigour", "8 independent source seeds per controller and task (160 policy branches, 20,000 steps each). Exact Wilcoxon signed-rank tests with Holm-Bonferroni correction.")
            ]
        },
        {
            "title": "Architectural Inductive Biases: Memory vs. Reactivity",
            "subtitle": "History-Conditioned Causal Transformer vs. Reactive MLP Baseline",
            "bullets": [
                ("Reactive Controller Baseline (MLP)", "Standard SAC policy: 2 hidden layers with 256 units each, ReLU activations. 71.9k actor / 72.5k critic parameters. Memoryless mapping s_t -> a_t, strictly reliant on instantaneous input."),
                ("History-Conditioned Controller (Transformer)", "GPT-style causal transformer: sliding context window K = 10 steps, token embedding d_model = 64, 2 causal self-attention blocks (4 heads), LayerNorm, and MLP projection head (87.6k actor / 92.3k critic parameters)."),
                ("In-Context Boundary Handling", "Context resets at episode boundaries. The model observes historical trajectory segments; cross-episode adaptation occurs exclusively through continued parameter updates."),
                ("Replay Mixing Rehearsal", "50% nominal clean transitions mixed with 50% online corrupted transitions to investigate whether memory preservation impairs adaptation velocity.")
            ]
        },
        {
            "title": "Empirical Findings: HalfCheetah-v5 Dynamics",
            "subtitle": "High-Frequency Noise Filtering vs. Catastrophic Observation Delay Collapse",
            "image": os.path.abspath("paper/fig_curves_hc.png"),
            "bullets": [
                ("Observation 1: Noise Resilience", "Under Gaussian sensor noise, both reactive MLP and transformer adapt swiftly, recovering near-optimal velocity. Replay mixing successfully prevents catastrophic forgetting with minimal penalty."),
                ("Observation 2: The Severe Delay Crash", "Observation delay causes immediate, severe performance collapse (>2,000 return penalty). Continued training achieves only partial recovery across 20,000 steps."),
                ("The Transformer Delay Paradox", "Under delay, the transformer actually struggled more than the reactive MLP! Causal self-attention attended to stale past tokens that misrepresented current plant physics, amplifying control latency.")
            ]
        },
        {
            "title": "Empirical Findings: Ant-v5 & Morphological Sensitivity",
            "subtitle": "Morphological Standing Posture and the Warm-Start Re-Optimization Dip",
            "image": os.path.abspath("paper/fig_curves_ant.png"),
            "bullets": [
                ("Observation 3: Morphological Immunity", "Under the standardized 400,000-step source recipe, all 8 Ant source seeds converged to an ultra-stable standing posture (mean forward velocity +/-0.007 m/s)."),
                ("Negligible Fault Sensitivity", "Because standing requires minimal dynamic velocity feedback, sensor noise and delay caused virtually zero return penalty. Reported transparently rather than retroactively tuning hyperparameters."),
                ("Observation 4: Re-Optimization Transient", "All Ant branches (including clean controls) dipped in return during the first 2,000 steps before recovering. Scientific Insight: This transient dip is caused by re-starting the Adam optimizer, NOT by sensor faults!")
            ]
        },
        {
            "title": "Mechanistic Critic Diagnostic: Soft Bellman Target Distortion",
            "subtitle": "Explaining Why Replay Buffer Mixing Hurt Adaptation Under Observation Delay",
            "bullets": [
                ("Diagnostic Framework", "Both soft Q-critics evaluated on held-out clean and corrupted trajectories at source checkpoint and after 20,000 steps. Evaluated soft Bellman residual and Monte-Carlo value-bias gap (Bias_corrupt - Bias_clean)."),
                ("Massive Source Value Overestimation", "Source critics severely overvalued delayed history states (+171 and +268 return units in HalfCheetah)."),
                ("Distortion From Replay Mixing", "Training on corrupted data alone shrank the bias gap to +27 and +114. Mixing 50% clean replay maintained the gap elevated at +89 and +204 (paired effect: +62 [16, 109] and +90 [41, 139], 7/8 seeds positive)."),
                ("Theoretical Explanation", "Mixing stale clean transitions creates an off-policy distribution mismatch that actively distorts the soft Bellman target, proving that naive rehearsal harms adaptation under non-Markovian delay.")
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
            "title": "Answers to the Research Questions: Key Takeaways",
            "subtitle": "Actionable Scientific Insights for Fault-Tolerant Reinforcement Learning",
            "bullets": [
                ("Answer to RQ1 (Adaptation vs Retention)", "Adaptation is fast and complete under high-frequency noise with full retention, but severely degraded under delay where nominal competence is permanently impaired."),
                ("Answer to RQ2 (Transformer Inductive Bias)", "History conditioning acts as a natural temporal low-pass filter under noise, but exacerbates vulnerability under delay unless delay is explicitly compensated in the state representation."),
                ("Answer to RQ3 (Replay Buffer Trade-off)", "Replay buffer mixing is not a universal panacea: it preserves nominal skills under noise, but induces severe soft Bellman target distortion under non-Markovian delay."),
                ("Answer to RQ4 (High-Rate Control Viability)", "History-conditioned transformers require only 87k parameters and execute in 0.58 ms on a single CPU thread, satisfying real-time 500 Hz robotics.")
            ]
        },
        {
            "title": "The Path Forward & Open Science",
            "subtitle": "Returning to TrackMania 2020, Physical Robotics, and Open Artifact Release",
            "bullets": [
                ("Implications for TrackMania & Robotics", "To overcome real-world delay, future controllers must incorporate explicit delay-aware predictive heads, temporal attention discounting, and dynamic uncertainty-weighted replay sampling."),
                ("Scalability Outlook", "Next phase investigates scaling these insights to high-dimensional continuous control (Humanoid-v5) and vision-based end-to-end TrackMania pipelines."),
                ("Full Open Science Release", "Complete reproducible codebase, sensor fault wrappers, frozen seed protocols, evaluation records, and checkpoints available at: https://github.com/Fane-Nathan/tmrl"),
                ("Primary Contact", "Felix Surjodinoto (felix.surjodinoto@binus.ac.id) | Bina Nusantara University")
            ]
        }
    ]

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
        
        for shp in list(slide.shapes):
            if shp.has_text_frame:
                txt = shp.text.strip()
                if "Title" == txt:
                    shp.text = ""
                    p = shp.text_frame.paragraphs[0]
                    p.text = cdata["title"]
                    p.font.name = "Montserrat"
                    p.font.size = Pt(28)
                    p.font.bold = True
                    p.font.color.rgb = RGBColor(0x00, 0x1E, 0x61)
                    shp.left = Pt(78)
                    shp.top = Pt(135)
                    shp.width = Pt(1150)
                    shp.height = Pt(45)
                elif "Subtitle" == txt:
                    shp.text = ""
                    p = shp.text_frame.paragraphs[0]
                    p.text = cdata["subtitle"]
                    p.font.name = "Montserrat"
                    p.font.size = Pt(19)
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

            img_path = cdata["image"]
            img_left = Pt(780)
            img_top = Pt(230)
            img_w = Pt(580)
            pic = slide.shapes.add_picture(img_path, img_left, img_top, width=img_w)
            max_h = Pt(500)
            if pic.height > max_h:
                scale = max_h / pic.height
                pic.width = int(pic.width * scale)
                pic.height = max_h
            pic.top = Pt(235 + (500 - pic.height / Pt(1)) / 2)

        elif has_table:
            t_rows = len(cdata["table_data"])
            t_cols = len(cdata["table_data"][0])
            tbl_shape = slide.shapes.add_table(t_rows, t_cols, Pt(78), Pt(225), Pt(1280), Pt(190))
            tbl = tbl_shape.table
            
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
                r1.font.size = Pt(16)
                r1.font.bold = True
                r1.font.color.rgb = RGBColor(0x00, 0x1E, 0x61)
                
                r2 = p.add_run()
                r2.text = desc
                r2.font.name = "Open Sans"
                r2.font.size = Pt(16)
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

    # Copy to target filenames
    shutil.copyfile(out_pdf, onedrive_pdf)
    shutil.copyfile(out_pdf, local_slides_pdf)
    print(f"Copied to: {onedrive_pdf}")
    print(f"Copied to: {local_slides_pdf}")

if __name__ == '__main__':
    build_presentation()
