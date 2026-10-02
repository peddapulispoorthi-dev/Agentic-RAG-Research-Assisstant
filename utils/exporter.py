import io
import logging

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

logger = logging.getLogger(__name__)

def export_to_markdown(report_text: str) -> str:
    """Format and validate Markdown report string for UI download."""
    if not report_text:
        return "# Deep Research Report\n\nNo content available for export."
    return report_text

def export_to_pdf(report_text: str) -> bytes:
    """Convert Markdown report string into a styled PDF document using ReportLab."""
    if not report_text:
        report_text = "# Deep Research Report\n\nNo content available."

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        'ReportH2',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=14,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'ReportBullet',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        leftIndent=15,
        textColor=colors.HexColor('#334155'),
        spaceAfter=4
    )

    story = []
    lines = report_text.split('\n')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
            continue
            
        # Clean HTML tags to prevent ReportLab XML parsing errors
        clean_text = stripped.replace('<', '&lt;').replace('>', '&gt;')
        
        # Convert simple markdown bold/italic if needed
        clean_text = clean_text.replace('**', '')

        if stripped.startswith('# '):
            heading_text = stripped[2:].replace('<', '&lt;').replace('>', '&gt;').replace('**', '')
            story.append(Paragraph(heading_text, title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0EA5E9'), spaceAfter=10))
        elif stripped.startswith('## '):
            heading_text = stripped[3:].replace('<', '&lt;').replace('>', '&gt;').replace('**', '')
            story.append(Paragraph(heading_text, h2_style))
        elif stripped.startswith('### '):
            heading_text = stripped[4:].replace('<', '&lt;').replace('>', '&gt;').replace('**', '')
            story.append(Paragraph(heading_text, h2_style))
        elif stripped.startswith('- ') or stripped.startswith('* '):
            bullet_text = stripped[2:].replace('<', '&lt;').replace('>', '&gt;').replace('**', '')
            story.append(Paragraph(f"• {bullet_text}", bullet_style))
        else:
            story.append(Paragraph(clean_text, body_style))

    try:
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"PDF generation failed in ReportLab: {e}")
        # Fallback simple text-based PDF canvas render
        from reportlab.pdfgen import canvas
        fallback_buffer = io.BytesIO()
        c = canvas.Canvas(fallback_buffer, pagesize=letter)
        y = 750
        c.setFont("Helvetica-Bold", 16)
        c.drawString(54, y, "Deep Research Report")
        y -= 30
        c.setFont("Helvetica", 10)
        for line in report_text.split('\n')[:50]:
            if y < 50:
                c.showPage()
                y = 750
                c.setFont("Helvetica", 10)
            c.drawString(54, y, line[:90])
            y -= 15
        c.save()
        fallback_buffer.seek(0)
        return fallback_buffer.getvalue()
