from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "reports" / "2026-04-13-pointnet2-attnv2-report-narrative.md"
OUT_DIR = ROOT / "docs" / "presentations"
OUT_PPTX = OUT_DIR / "pointnet2_attnv2_light_theme.pptx"
OUT_SCRIPT = OUT_DIR / "pointnet2_attnv2_presentation_script.md"


NAVY = RGBColor(18, 39, 71)
BLUE = RGBColor(0, 120, 212)
CYAN = RGBColor(81, 151, 255)
PALE_BLUE = RGBColor(233, 242, 251)
PALE_BLUE_2 = RGBColor(244, 248, 252)
SLATE = RGBColor(87, 102, 122)
DARK = RGBColor(35, 50, 72)
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(25, 135, 84)
RED = RGBColor(192, 57, 43)
AMBER = RGBColor(215, 152, 52)
BORDER = RGBColor(216, 226, 236)

TITLE_FONT = "Avenir Next"
BODY_FONT = "Avenir Next"


def set_slide_bg(slide, color=WHITE):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_top_bar(slide, color=BLUE):
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.18)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def add_footer(slide, page_num, total, label="PointNet++ Attention Enhancement"):
    line = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0.6), Inches(7.06), Inches(12.1), Inches(0.01)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER
    line.line.fill.background()

    text = slide.shapes.add_textbox(Inches(0.7), Inches(7.08), Inches(8.0), Inches(0.25))
    tf = text.text_frame
    p = tf.paragraphs[0]
    p.text = label
    p.font.name = BODY_FONT
    p.font.size = Pt(10)
    p.font.color.rgb = SLATE

    num = slide.shapes.add_textbox(Inches(12.1), Inches(7.05), Inches(0.6), Inches(0.25))
    tf = num.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    p.text = f"{page_num}/{total}"
    p.font.name = BODY_FONT
    p.font.size = Pt(10)
    p.font.color.rgb = SLATE


def add_title(slide, title, subtitle=None, section=None):
    add_top_bar(slide)
    if section:
        badge = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(0.65), Inches(0.45), Inches(1.8), Inches(0.42)
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = PALE_BLUE
        badge.line.color.rgb = PALE_BLUE
        tf = badge.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.text = section
        p.alignment = PP_ALIGN.CENTER
        p.font.name = BODY_FONT
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = BLUE

    tb = slide.shapes.add_textbox(Inches(0.7), Inches(0.95), Inches(12.0), Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = TITLE_FONT
    p.font.bold = True
    p.font.size = Pt(30)
    p.font.color.rgb = NAVY

    if subtitle:
        sb = slide.shapes.add_textbox(Inches(0.72), Inches(1.62), Inches(11.2), Inches(0.5))
        tf = sb.text_frame
        p = tf.paragraphs[0]
        p.text = subtitle
        p.font.name = BODY_FONT
        p.font.size = Pt(14)
        p.font.color.rgb = SLATE


def add_body_text(slide, left, top, width, height, paragraphs, font_size=18, color=DARK):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(4)
    tf.margin_right = Pt(4)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.clear()
    for idx, item in enumerate(paragraphs):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = item
        p.font.name = BODY_FONT
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.space_after = Pt(8)
        p.line_spacing = 1.18
    return box


def add_card(slide, left, top, width, height, title, lines, accent=BLUE, fill=WHITE, title_size=18, body_size=14):
    card = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = fill
    card.line.color.rgb = BORDER
    card.line.width = Pt(1.0)

    accent_bar = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, left, top, width, Inches(0.08))
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = accent
    accent_bar.line.fill.background()

    title_box = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.16), width - Inches(0.36), Inches(0.45))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = TITLE_FONT
    p.font.bold = True
    p.font.size = Pt(title_size)
    p.font.color.rgb = NAVY

    body_box = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.56), width - Inches(0.36), height - Inches(0.72))
    tf = body_box.text_frame
    tf.word_wrap = True
    tf.clear()
    for idx, line in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = BODY_FONT
        p.font.size = Pt(body_size)
        p.font.color.rgb = DARK
        p.space_after = Pt(6)
        p.line_spacing = 1.15
    return card


def add_metric_card(slide, left, top, width, height, value, label, accent=BLUE, delta=None):
    card = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = PALE_BLUE_2
    card.line.color.rgb = BORDER
    card.line.width = Pt(1.0)

    vb = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.15), width - Inches(0.36), Inches(0.52))
    tf = vb.text_frame
    p = tf.paragraphs[0]
    p.text = value
    p.font.name = TITLE_FONT
    p.font.bold = True
    p.font.size = Pt(28)
    p.font.color.rgb = accent

    lb = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.72), width - Inches(0.36), Inches(0.38))
    tf = lb.text_frame
    p = tf.paragraphs[0]
    p.text = label
    p.font.name = BODY_FONT
    p.font.size = Pt(13)
    p.font.color.rgb = DARK

    if delta:
        db = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(1.07), width - Inches(0.36), Inches(0.28))
        tf = db.text_frame
        p = tf.paragraphs[0]
        p.text = delta
        p.font.name = BODY_FONT
        p.font.size = Pt(12)
        p.font.color.rgb = GREEN if delta.startswith("+") else RED


def add_table(slide, left, top, width, height, columns, rows, header_fill=NAVY):
    shape = slide.shapes.add_table(len(rows) + 1, len(columns), left, top, width, height)
    table = shape.table
    table.first_row = True
    for idx, col in enumerate(columns):
        cell = table.cell(0, idx)
        cell.text = col
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        for p in cell.text_frame.paragraphs:
            p.font.name = BODY_FONT
            p.font.bold = True
            p.font.size = Pt(11)
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER

    for r_idx, row in enumerate(rows, start=1):
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r_idx % 2 else PALE_BLUE_2
            for p in cell.text_frame.paragraphs:
                p.font.name = BODY_FONT
                p.font.size = Pt(10.5)
                p.font.color.rgb = DARK
                p.alignment = PP_ALIGN.CENTER
    return table


def add_bar_chart(slide, left, top, width, height, categories, series):
    chart_data = CategoryChartData()
    chart_data.categories = categories
    for name, values in series:
        chart_data.add_series(name, values)

    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, left, top, width, height, chart_data
    ).chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.value_axis.maximum_scale = 100
    chart.value_axis.minimum_scale = 80
    chart.value_axis.tick_labels.font.size = Pt(10)
    chart.category_axis.tick_labels.font.size = Pt(10)
    chart.chart_title.has_text_frame = False

    colors = [BLUE, CYAN]
    for idx, series_obj in enumerate(chart.series):
        fill = series_obj.format.fill
        fill.solid()
        fill.fore_color.rgb = colors[idx % len(colors)]
    return chart


def add_line_chart(slide, left, top, width, height, categories, series):
    chart_data = CategoryChartData()
    chart_data.categories = categories
    for name, values in series:
        chart_data.add_series(name, values)

    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.LINE_MARKERS, left, top, width, height, chart_data
    ).chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.value_axis.maximum_scale = 100
    chart.value_axis.minimum_scale = 30
    chart.value_axis.tick_labels.font.size = Pt(10)
    chart.category_axis.tick_labels.font.size = Pt(10)
    colors = [NAVY, BLUE]
    for idx, series_obj in enumerate(chart.series):
        line = series_obj.format.line
        line.color.rgb = colors[idx % len(colors)]
        line.width = Pt(2.5)
    return chart


def add_timeline(slide, steps):
    y = Inches(4.7)
    line = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0.9), y, Inches(11.5), Inches(0.04))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER
    line.line.fill.background()

    x_positions = [0.95, 3.1, 5.25, 7.4, 9.55, 11.7]
    for idx, step in enumerate(steps):
        cx = Inches(x_positions[idx])
        circle = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, cx, y - Inches(0.14), Inches(0.26), Inches(0.26))
        circle.fill.solid()
        circle.fill.fore_color.rgb = BLUE if idx >= 2 else CYAN
        circle.line.fill.background()
        label = slide.shapes.add_textbox(cx - Inches(0.1), y + Inches(0.18), Inches(1.2), Inches(0.8))
        tf = label.text_frame
        p = tf.paragraphs[0]
        p.text = step
        p.font.name = BODY_FONT
        p.font.size = Pt(11)
        p.font.color.rgb = DARK
        p.alignment = PP_ALIGN.CENTER


def write_script(path):
    content = """# PointNet++ Attention Enhancement Presentation Script

## Slide 1 - Title
Today I will present our PointNet++ attention enhancement project. The core question behind this work was whether attention-based modifications could improve PointNet++ single-scale grouping classification, especially on ModelNet40, while still remaining usable on Apple Silicon hardware. The work evolved beyond a simple model tweak and became a combined architecture-and-infrastructure project.

## Slide 2 - Project Scope
The project had two parallel goals. One goal was engineering stability. The original repository needed to be hardened for Apple Silicon and MPS so that later experimental results would be trustworthy. The second goal was architectural exploration. We wanted to see whether channel attention, local attention pooling, and cross-level fusion could improve PointNet++ classification behavior.

## Slide 3 - Codebase Evolution
The work happened across four main branches. The upstream baseline stayed on master. The Apple Silicon branch made the repository train reliably on MPS and introduced the first SE and attention models. The attention-v2 branch implemented a corrected attention architecture. The final branch extended the repository to 3D MNIST and upgraded the training and evaluation pipeline with persistent experiment records.

## Slide 4 - Baseline Method
The baseline model is the PointNet++ SSG classifier. It uses three set abstraction stages to move from local neighborhoods to a global feature, and then classifies from a 1024-dimensional descriptor. This baseline is already strong, which is important because any new method has to beat a competent local retrain, not just the original 2017 paper number.

## Slide 5 - Optimization Path
The architectural path was baseline, then SE, then attention-v1, then attention-v2. The SE model added channel recalibration and a stronger classifier. Attention-v1 introduced local attention pooling and cross-level fusion. Attention-v2 was not just another variant. It was specifically designed to correct the places where attention-v1 was technically unsound.

## Slide 6 - Why SE Was Added
The SE version applies Squeeze-and-Excitation after the first two abstraction layers. The principle is that not every feature channel is equally useful, so the network should learn which channels to emphasize and which to suppress. In practice, this improved class accuracy, which suggests better class balance, but it did not surpass the baseline on top-line instance accuracy.

## Slide 7 - Why Attention v1 Failed
Attention-v1 failed for three main reasons. First, PointNet++ radius grouping duplicates the first valid neighbor to fill padded slots, which is safe for max pooling but corrupts softmax attention. Second, the cross-level fusion compressed too much information before attention had any chance to help. Third, the classifier head was effectively too weak. These problems explain why the first attention model fell well below baseline.

## Slide 8 - Attention v2 Design
Attention-v2 fixed those issues directly. It introduced masked local attention so softmax is applied only over valid neighbors. It fused attention pooling with max pooling so the model keeps the robustness of PointNet++ while adding learnable local weighting. It also used a stronger cross-level fusion design with mean and max summaries, a learnable CLS token, and a residual path from the global feature. Finally, it used a stronger nonlinear classifier head.

## Slide 9 - Engineering and Logging
The final pipeline also improved the surrounding experiment system. Device selection is now unified across MPS, CUDA, and CPU. The training script supports multiple datasets through a shared interface. Every run stores configuration, per-epoch metrics, and a final summary JSON. This matters because later analysis and presentation depend on reproducible artifacts, not only console logs.

## Slide 10 - Experimental Setup
For the main classification runs, we used 1024 input points, Adam optimizer, an initial learning rate of 0.001, StepLR scheduling, and no normal vectors for the principal comparisons. On ModelNet40, the runs were informative but not perfectly controlled because the batch size differed between some variants. On 3D MNIST, the baseline and attention-v2 comparison used exactly the same setup, making that cross-dataset comparison cleaner.

## Slide 11 - ModelNet40 Results
On ModelNet40, the strongest local baseline reached 92.459 percent instance accuracy and 89.175 percent class accuracy. The SE model improved class accuracy slightly but not instance accuracy. Attention-v1 underperformed clearly. Attention-v2 recovered most of that lost ground and improved strongly over attention-v1, especially on class accuracy, but it still did not surpass the baseline on instance accuracy. So the final answer on ModelNet40 is mixed: attention-v2 is much better than v1, but baseline remains best on the primary metric.

## Slide 12 - Comparison with the PointNet++ Paper
The original PointNet++ paper reported 90.7 percent without normals and 91.9 percent with normals on ModelNet40. Our local baseline is stronger than both of those numbers even without normals, which shows that the modern codebase and training setup already form a stronger baseline than the original paper configuration. That means the correct comparison target for our new method is the local retrain, not just the paper result.

## Slide 13 - 3D MNIST Integration
To verify the method on another dataset, we integrated 3D MNIST. This dataset was chosen because it was available locally in point-cloud-oriented HDF5 files and could be added to the repository quickly. A dedicated dataloader was written to sample each object down to 1024 points, normalize coordinates, and optionally include normals. The pipeline also forces zero dataloader workers on macOS for HDF5 safety.

## Slide 14 - 3D MNIST Results
On 3D MNIST, the baseline achieved 98.611 percent instance accuracy and 98.623 percent class accuracy. Attention-v2 slightly outperformed it with 98.710 percent instance accuracy and 98.939 percent class accuracy. The gain is small, but it is consistent and appears more clearly in class accuracy. The convergence curves also show that the baseline learns faster early, while attention-v2 improves more gradually and ends slightly higher.

## Slide 15 - Drawbacks and Future Work
The main limitation is that attention-v2 does not beat the strongest baseline on ModelNet40 instance accuracy. The second limitation is efficiency. Attention-v2 is much larger than the baseline. There is also no full ablation study and no multi-seed evaluation. Finally, while 3D MNIST is a valid second classification dataset, it is not as strong a realism benchmark as ScanObjectNN. The most valuable next steps would be controlled same-batch-size reruns, lighter attention-v2 blocks, full ablation, and evaluation on a stronger second dataset.

## Slide 16 - Final Conclusion
The project produced a stable Apple Silicon training pipeline, a corrected attention architecture, and a reusable cross-dataset evaluation workflow. The scientific result is honest and balanced. Attention-v2 is clearly better than attention-v1 and shows some promise on a second dataset, but it does not replace the strong PointNet++ SSG baseline on ModelNet40. That balance gives us a credible presentation story built around design, diagnosis, evidence, and limitations.
"""
    path.write_text(content)


def build_presentation():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Source report not found: {REPORT_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    total_slides = 16

    # Slide 1
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)
    hero = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(8.9), Inches(-0.6), Inches(5.2), Inches(5.2))
    hero.fill.solid()
    hero.fill.fore_color.rgb = PALE_BLUE
    hero.line.fill.background()
    hero2 = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(10.8), Inches(4.8), Inches(3.0), Inches(3.0))
    hero2.fill.solid()
    hero2.fill.fore_color.rgb = PALE_BLUE_2
    hero2.line.fill.background()
    add_top_bar(slide, BLUE)
    title = slide.shapes.add_textbox(Inches(0.75), Inches(1.05), Inches(8.3), Inches(1.6))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "PointNet++ Attention Enhancement"
    p.font.name = TITLE_FONT
    p.font.bold = True
    p.font.size = Pt(34)
    p.font.color.rgb = NAVY
    p = tf.add_paragraph()
    p.text = "ModelNet40 and 3D MNIST Classification Study"
    p.font.name = TITLE_FONT
    p.font.size = Pt(34)
    p.font.color.rgb = NAVY
    p.font.bold = True
    sub = slide.shapes.add_textbox(Inches(0.8), Inches(2.85), Inches(7.0), Inches(1.0))
    tf = sub.text_frame
    p = tf.paragraphs[0]
    p.text = "Light theme presentation generated from the narrative project report"
    p.font.name = BODY_FONT
    p.font.size = Pt(18)
    p.font.color.rgb = SLATE
    p = tf.add_paragraph()
    p.text = "Style: Corporate Professional, adapted for a clean academic presentation"
    p.font.name = BODY_FONT
    p.font.size = Pt(16)
    p.font.color.rgb = SLATE
    add_metric_card(slide, Inches(0.78), Inches(4.15), Inches(2.2), Inches(1.45), "92.459%", "Best local baseline on ModelNet40", NAVY)
    add_metric_card(slide, Inches(3.18), Inches(4.15), Inches(2.2), Inches(1.45), "92.056%", "Best attn-v2 on ModelNet40", BLUE)
    add_metric_card(slide, Inches(5.58), Inches(4.15), Inches(2.2), Inches(1.45), "98.710%", "Best attn-v2 on 3D MNIST", GREEN)
    src = slide.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(9.2), Inches(0.4))
    tf = src.text_frame
    p = tf.paragraphs[0]
    p.text = f"Source report: {REPORT_PATH.relative_to(ROOT)}"
    p.font.name = BODY_FONT
    p.font.size = Pt(11)
    p.font.color.rgb = SLATE
    add_footer(slide, 1, total_slides)

    # Slide 2
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Project Scope", "Two parallel threads shaped the final outcome", "Context")
    add_card(slide, Inches(0.8), Inches(2.0), Inches(3.8), Inches(3.2), "Engineering Track", [
        "Stabilize training on Apple Silicon and MPS.",
        "Harden PointNet++ utilities for safer indexing, masking, and dataloader behavior.",
        "Build a reusable logging pipeline with structured experiment records."
    ], accent=BLUE, fill=PALE_BLUE_2)
    add_card(slide, Inches(4.8), Inches(2.0), Inches(3.8), Inches(3.2), "Architecture Track", [
        "Start from PointNet++ SSG as the baseline.",
        "Explore SE blocks, local attention pooling, and cross-level fusion.",
        "Diagnose the failure of attention v1 and redesign it into attention v2."
    ], accent=CYAN, fill=PALE_BLUE_2)
    add_card(slide, Inches(8.8), Inches(2.0), Inches(3.7), Inches(3.2), "Outcome", [
        "A stable Mac-compatible codebase.",
        "A corrected attention model that is much stronger than the first attempt.",
        "A second-dataset evaluation pipeline on 3D MNIST."
    ], accent=GREEN, fill=PALE_BLUE_2)
    add_footer(slide, 2, total_slides)

    # Slide 3
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Codebase Evolution", "The project progressed through four branches", "Timeline")
    add_body_text(slide, Inches(0.8), Inches(2.0), Inches(12.0), Inches(0.7), [
        "The work moved from baseline reproduction to platform stabilization, then to corrected architecture design, and finally to cross-dataset validation."
    ], font_size=17)
    add_timeline(slide, [
        "master\nUpstream",
        "feat/apple_silicon\nMPS support",
        "SE\nChannel attention",
        "attn-v1\nFirst attention",
        "attn-v2\nCorrected design",
        "3D MNIST\nCross-dataset",
    ])
    add_card(slide, Inches(0.9), Inches(5.45), Inches(3.55), Inches(1.15), "Key branch", [
        "feat/apple_silicon"
    ], accent=BLUE, fill=WHITE, title_size=16, body_size=18)
    add_card(slide, Inches(4.9), Inches(5.45), Inches(3.55), Inches(1.15), "Key branch", [
        "codex/pointnet2-attn-v2-mac-safe"
    ], accent=CYAN, fill=WHITE, title_size=16, body_size=18)
    add_card(slide, Inches(8.9), Inches(5.45), Inches(3.55), Inches(1.15), "Key branch", [
        "codex/3dmnist-attnv2-training"
    ], accent=GREEN, fill=WHITE, title_size=16, body_size=18)
    add_footer(slide, 3, total_slides)

    # Slide 4
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Baseline Method", "PointNet++ SSG classification pipeline", "Architecture")
    add_card(slide, Inches(0.85), Inches(2.1), Inches(2.4), Inches(2.2), "SA1", [
        "512 points",
        "radius 0.2",
        "32 neighbors",
        "64 -> 64 -> 128"
    ], accent=BLUE, fill=PALE_BLUE_2)
    add_card(slide, Inches(3.55), Inches(2.1), Inches(2.4), Inches(2.2), "SA2", [
        "128 points",
        "radius 0.4",
        "64 neighbors",
        "128 -> 128 -> 256"
    ], accent=CYAN, fill=PALE_BLUE_2)
    add_card(slide, Inches(6.25), Inches(2.1), Inches(2.4), Inches(2.2), "SA3", [
        "Global grouping",
        "256 -> 512 -> 1024",
        "single global descriptor"
    ], accent=BLUE, fill=PALE_BLUE_2)
    add_card(slide, Inches(8.95), Inches(2.1), Inches(2.5), Inches(2.2), "Classifier", [
        "1024 -> 512 -> 256 -> class",
        "standard PointNet++ SSG head"
    ], accent=NAVY, fill=PALE_BLUE_2)
    add_body_text(slide, Inches(0.95), Inches(4.9), Inches(11.5), Inches(1.3), [
        "This baseline is already strong on ModelNet40, so the real target is not only to beat the original paper, but to beat a strong local retrain."
    ], font_size=18)
    add_footer(slide, 4, total_slides)

    # Slide 5
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Optimization Path", "What each model variant tried to improve", "Methods")
    add_card(slide, Inches(0.8), Inches(2.0), Inches(2.9), Inches(3.6), "Baseline", [
        "Pure PointNet++ SSG.",
        "Strong local hierarchy.",
        "Reference model for all later comparisons."
    ], accent=NAVY, fill=WHITE)
    add_card(slide, Inches(3.95), Inches(2.0), Inches(2.9), Inches(3.6), "SE Version", [
        "Add channel attention after SA1 and SA2.",
        "Deepen the classifier.",
        "Goal: better channel discrimination and class balance."
    ], accent=BLUE, fill=WHITE)
    add_card(slide, Inches(7.1), Inches(2.0), Inches(2.9), Inches(3.6), "Attention v1", [
        "Replace max pooling with local attention pooling.",
        "Fuse l1, l2, l3 with cross-level attention.",
        "Goal: richer local weighting and multi-scale fusion."
    ], accent=CYAN, fill=WHITE)
    add_card(slide, Inches(10.25), Inches(2.0), Inches(2.25), Inches(3.6), "Attention v2", [
        "Masked local attention.",
        "Hybrid attention + max pooling.",
        "CLS-token fusion plus global residual.",
        "Stronger nonlinear classifier."
    ], accent=GREEN, fill=WHITE)
    add_footer(slide, 5, total_slides)

    # Slide 6
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Why SE Was Introduced", "Channel recalibration plus a stronger head", "SE")
    add_body_text(slide, Inches(0.8), Inches(2.0), Inches(5.2), Inches(3.0), [
        "SE stands for Squeeze-and-Excitation. The block globally averages each feature channel, learns a channel importance vector, and rescales the original features.",
        "The intuition is simple: not every feature channel contributes equally to classification, so the network should emphasize informative channels and suppress weaker ones."
    ], font_size=18)
    add_metric_card(slide, Inches(7.0), Inches(2.0), Inches(2.25), Inches(1.45), "91.976%", "SE best instance acc", BLUE)
    add_metric_card(slide, Inches(9.5), Inches(2.0), Inches(2.25), Inches(1.45), "89.626%", "SE best class acc", GREEN)
    add_body_text(slide, Inches(6.95), Inches(3.85), Inches(5.0), Inches(1.3), [
        "Result: SE improved class accuracy over baseline, but it did not surpass baseline on instance accuracy."
    ], font_size=17)
    add_footer(slide, 6, total_slides)

    # Slide 7
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Why Attention v1 Failed", "The first attention design was conceptually interesting but technically flawed", "Diagnosis")
    add_card(slide, Inches(0.8), Inches(2.0), Inches(3.75), Inches(3.4), "Problem 1: invalid local attention", [
        "PointNet++ radius grouping duplicates the first valid neighbor when padding.",
        "This is safe for max pooling.",
        "It is not safe for softmax attention because fake duplicates collect probability mass."
    ], accent=RED, fill=WHITE)
    add_card(slide, Inches(4.8), Inches(2.0), Inches(3.75), Inches(3.4), "Problem 2: over-compressed fusion", [
        "Each hierarchy level was reduced to a single mean-pooled token.",
        "The attended tokens were then averaged again.",
        "Too much structure was discarded before attention could help."
    ], accent=AMBER, fill=WHITE)
    add_card(slide, Inches(8.8), Inches(2.0), Inches(3.75), Inches(3.4), "Problem 3: weak classifier", [
        "The post-fusion classifier behaved too close to a shallow linear head.",
        "There was not enough nonlinear capacity to exploit the fused representation."
    ], accent=BLUE, fill=WHITE)
    add_metric_card(slide, Inches(4.8), Inches(5.7), Inches(3.75), Inches(1.0), "91.392% / 86.231%", "Best ModelNet40 result of attention v1", RED)
    add_footer(slide, 7, total_slides)

    # Slide 8
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Attention v2 Design", "A corrected architecture, not just a larger one", "attn-v2")
    add_card(slide, Inches(0.8), Inches(2.0), Inches(2.9), Inches(3.2), "Masked local attention", [
        "Return explicit validity masks from grouping.",
        "Normalize softmax only over valid neighbors."
    ], accent=BLUE, fill=PALE_BLUE_2)
    add_card(slide, Inches(3.95), Inches(2.0), Inches(2.9), Inches(3.2), "Hybrid local pooling", [
        "Fuse attention pooling with max pooling.",
        "Keep PointNet++ robustness while adding learnable weighting."
    ], accent=CYAN, fill=PALE_BLUE_2)
    add_card(slide, Inches(7.1), Inches(2.0), Inches(2.9), Inches(3.2), "Cross-level fusion", [
        "Use mean and max summaries.",
        "Add a CLS token and a global residual path from l3."
    ], accent=GREEN, fill=PALE_BLUE_2)
    add_card(slide, Inches(10.25), Inches(2.0), Inches(2.25), Inches(3.2), "Classifier", [
        "Wider nonlinear head.",
        "512 -> 256 -> 128 -> class."
    ], accent=NAVY, fill=PALE_BLUE_2)
    add_body_text(slide, Inches(0.9), Inches(5.55), Inches(11.6), Inches(0.8), [
        "Result: attention v2 recovered most of the performance lost by attention v1 and strongly improved class balance."
    ], font_size=17)
    add_footer(slide, 8, total_slides)

    # Slide 9
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Engineering and Logging Infrastructure", "The project became reproducible, not just runnable", "Pipeline")
    add_card(slide, Inches(0.8), Inches(2.0), Inches(3.85), Inches(3.6), "Platform support", [
        "Automatic MPS / CUDA / CPU device selection.",
        "Safer index handling and masking on Apple Silicon.",
        "Conservative worker settings for macOS."
    ], accent=BLUE, fill=WHITE)
    add_card(slide, Inches(4.8), Inches(2.0), Inches(3.85), Inches(3.6), "Dataset-aware training", [
        "A unified train script now supports ModelNet40, ModelNet10, and 3D MNIST.",
        "The evaluation script can reuse run configuration automatically."
    ], accent=CYAN, fill=WHITE)
    add_card(slide, Inches(8.8), Inches(2.0), Inches(3.7), Inches(3.6), "Persistent experiment records", [
        "run_config.json for exact arguments and environment.",
        "metrics.csv for per-epoch curves.",
        "summary.json for the best final result.",
        "Exact code snapshots copied into each run directory."
    ], accent=GREEN, fill=WHITE)
    add_footer(slide, 9, total_slides)

    # Slide 10
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Experimental Setup and Metrics", "What was measured and how", "Evaluation")
    add_card(slide, Inches(0.8), Inches(2.0), Inches(3.8), Inches(3.3), "Shared setup", [
        "1024 input points.",
        "Adam optimizer, learning rate 0.001, weight decay 1e-4.",
        "StepLR with step size 20 and gamma 0.7.",
        "Primary runs used no normals."
    ], accent=BLUE, fill=WHITE)
    add_card(slide, Inches(4.8), Inches(2.0), Inches(3.7), Inches(3.3), "Metrics", [
        "Instance accuracy: percentage of correctly classified test samples.",
        "Class accuracy: average of per-class accuracies.",
        "Train accuracy, learning rate, and global step logged per epoch."
    ], accent=CYAN, fill=WHITE)
    add_card(slide, Inches(8.85), Inches(2.0), Inches(3.55), Inches(3.3), "Important caveat", [
        "ModelNet40 variants did not all use the same batch size.",
        "3D MNIST baseline and attn-v2 were run under the same settings, making that comparison cleaner."
    ], accent=AMBER, fill=WHITE)
    add_footer(slide, 10, total_slides)

    # Slide 11
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "ModelNet40 Results", "The local baseline remains the strongest top-line model", "Results")
    add_table(
        slide,
        Inches(0.7),
        Inches(1.9),
        Inches(6.1),
        Inches(2.8),
        ["Model", "Best Inst.", "Best Class", "Best Epoch"],
        [
            ["Baseline", "92.459%", "89.175%", "68"],
            ["SE", "91.976%", "89.626%", "92"],
            ["attn-v1", "91.392%", "86.231%", "88"],
            ["attn-v2", "92.056%", "89.597%", "100"],
        ],
    )
    add_bar_chart(
        slide,
        Inches(7.1),
        Inches(2.0),
        Inches(5.4),
        Inches(3.5),
        ["Baseline", "SE", "attn-v1", "attn-v2"],
        [("Instance Accuracy", [92.459, 91.976, 91.392, 92.056]), ("Class Accuracy", [89.175, 89.626, 86.231, 89.597])],
    )
    add_body_text(slide, Inches(0.8), Inches(5.9), Inches(11.6), Inches(0.6), [
        "Attention v2 is a strong repair over attention v1, but it still trails the local baseline by 0.403 points in instance accuracy."
    ], font_size=16)
    add_footer(slide, 11, total_slides)

    # Slide 12
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Comparison with the PointNet++ Paper", "The local retrain is the real benchmark", "Paper")
    add_table(
        slide,
        Inches(0.8),
        Inches(2.0),
        Inches(6.0),
        Inches(2.8),
        ["Method", "Features", "Instance Acc."],
        [
            ["PointNet++ paper", "XYZ only", "90.7%"],
            ["PointNet++ paper", "XYZ + normals", "91.9%"],
            ["Local baseline", "XYZ only", "92.459%"],
            ["attn-v2", "XYZ only", "92.056%"],
        ],
    )
    add_metric_card(slide, Inches(7.15), Inches(2.05), Inches(2.45), Inches(1.5), "+1.759", "Baseline vs paper XYZ", GREEN)
    add_metric_card(slide, Inches(9.95), Inches(2.05), Inches(2.45), Inches(1.5), "+1.357", "attn-v2 vs paper XYZ", BLUE)
    add_body_text(slide, Inches(7.1), Inches(4.0), Inches(5.2), Inches(1.8), [
        "The codebase and training recipe already outperform the original 2017 paper result. That is why the local baseline, not the paper alone, is the correct target for judging the new method."
    ], font_size=17)
    add_footer(slide, 12, total_slides)

    # Slide 13
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "3D MNIST Integration", "A practical second dataset for cross-dataset validation", "Dataset")
    add_card(slide, Inches(0.8), Inches(2.0), Inches(3.8), Inches(3.4), "Why 3D MNIST", [
        "It was available locally as point-cloud-oriented HDF5 files.",
        "It could be integrated much faster than ScanObjectNN or raw ShapeNetCore.",
        "It still provides a real second 3D classification task."
    ], accent=BLUE, fill=WHITE)
    add_card(slide, Inches(4.8), Inches(2.0), Inches(3.8), Inches(3.4), "What the loader does", [
        "Read HDF5 groups with points, normals, and labels.",
        "Sample each object down to 1024 points.",
        "Normalize XYZ and optionally concatenate normals."
    ], accent=CYAN, fill=WHITE)
    add_card(slide, Inches(8.8), Inches(2.0), Inches(3.7), Inches(3.4), "Dataset properties", [
        "Train: 5000 samples.",
        "Test: 1000 samples.",
        "Classes: 10.",
        "Average raw point count above 20k."
    ], accent=GREEN, fill=WHITE)
    add_footer(slide, 13, total_slides)

    # Slide 14
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "3D MNIST Results", "A clean same-setting comparison between baseline and attn-v2", "Results")
    add_metric_card(slide, Inches(0.8), Inches(2.0), Inches(2.6), Inches(1.6), "98.611%", "Baseline best instance acc", NAVY)
    add_metric_card(slide, Inches(3.6), Inches(2.0), Inches(2.6), Inches(1.6), "98.623%", "Baseline best class acc", NAVY)
    add_metric_card(slide, Inches(6.55), Inches(2.0), Inches(2.6), Inches(1.6), "98.710%", "attn-v2 best instance acc", BLUE, delta="+0.099 vs baseline")
    add_metric_card(slide, Inches(9.35), Inches(2.0), Inches(2.6), Inches(1.6), "98.939%", "attn-v2 best class acc", GREEN, delta="+0.316 vs baseline")
    add_line_chart(
        slide,
        Inches(0.95),
        Inches(4.0),
        Inches(6.0),
        Inches(2.5),
        ["1", "5", "10", "20", "40", "60", "80", "100"],
        [
            ("Baseline", [71.726, 87.599, 83.631, 94.246, 96.627, 98.611, 98.016, 98.611]),
            ("attn-v2", [37.798, 89.683, 89.583, 95.139, 97.718, 98.313, 98.115, 98.512]),
        ],
    )
    add_body_text(slide, Inches(7.35), Inches(4.15), Inches(5.0), Inches(2.0), [
        "Baseline learns faster at the start.",
        "attn-v2 catches up gradually and ends slightly stronger on both best metrics.",
        "The gain is small, but it is consistent and class-aware."
    ], font_size=17)
    add_footer(slide, 14, total_slides)

    # Slide 15
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Drawbacks and Future Work", "The final result is credible because the limitations are explicit", "Limitations")
    add_card(slide, Inches(0.8), Inches(2.0), Inches(5.4), Inches(3.6), "Drawbacks", [
        "attn-v2 does not beat the strongest local baseline on ModelNet40 instance accuracy.",
        "The parameter count is very high: 5.95M vs 1.48M for baseline.",
        "No full ablation and no multi-seed evaluation were run.",
        "3D MNIST is useful but not a strong realism benchmark."
    ], accent=RED, fill=WHITE)
    add_card(slide, Inches(6.5), Inches(2.0), Inches(5.0), Inches(3.6), "Recommended next steps", [
        "Run same-batch-size controlled reruns on ModelNet40.",
        "Ablate masked attention, hybrid pooling, cross-level fusion, and classifier depth separately.",
        "Reduce the late-stage parameter cost in attention v2.",
        "Evaluate on a stronger second dataset such as ScanObjectNN."
    ], accent=GREEN, fill=WHITE)
    add_footer(slide, 15, total_slides)

    # Slide 16
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)
    ring = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(9.0), Inches(0.85), Inches(3.2), Inches(3.2))
    ring.fill.solid()
    ring.fill.fore_color.rgb = PALE_BLUE
    ring.line.fill.background()
    add_top_bar(slide, NAVY)
    add_title(slide, "Final Conclusion", "A balanced result is still a strong research outcome", "Takeaway")
    add_body_text(slide, Inches(0.8), Inches(2.05), Inches(7.2), Inches(2.9), [
        "The project delivered a stable Apple Silicon pipeline, a corrected attention model, and a reusable cross-dataset evaluation workflow.",
        "Attention v2 is clearly better than attention v1 and shows modest promise on a second dataset.",
        "However, the strongest PointNet++ SSG baseline remains the best ModelNet40 model in this project."
    ], font_size=19)
    add_metric_card(slide, Inches(8.25), Inches(2.1), Inches(3.4), Inches(1.55), "Honest result", "Better engineering and better diagnosis than a simple accuracy win", NAVY)
    add_metric_card(slide, Inches(8.25), Inches(3.95), Inches(3.4), Inches(1.55), "Presentation message", "A full research cycle: hypothesis, failure analysis, redesign, validation", BLUE)
    add_footer(slide, 16, total_slides)

    prs.save(OUT_PPTX)
    write_script(OUT_SCRIPT)


if __name__ == "__main__":
    build_presentation()
