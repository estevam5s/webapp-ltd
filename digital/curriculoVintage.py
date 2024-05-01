from pathlib import Path
import streamlit as st
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def curriculo():
    # --- PATH SETTINGS ---
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    css_file = current_dir / "styles" / "main.css"
    profile_pic_path = current_dir / "assets" / "profile-pic.png"

    # --- GENERAL SETTINGS ---
    PAGE_TITLE = "Digital CV | Your Name"
    PAGE_ICON = ":wave:"
    DESCRIPTION = """
    Describe yourself here...
    """

    # st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

    # --- HERO SECTION ---
    st.title("Digital CV")
    st.markdown("---")

    # Profile Picture
    profile_pic = st.file_uploader("Upload Profile Picture", type=["png", "jpg", "jpeg"])
    if profile_pic is not None:
        image = Image.open(profile_pic)
        st.image(image, caption="Uploaded Profile Picture", use_column_width=True)

    # Name
    name = st.text_input("Name", "Your Name")

    # Description
    description = st.text_area("Description", DESCRIPTION)

    # Email
    email = st.text_input("Email", "youremail@example.com")

    # Social Media
    st.subheader("Social Media")
    social_media = {}
    for platform in ["YouTube", "LinkedIn", "GitHub", "Twitter"]:
        social_media[platform] = st.text_input(platform, f"https://{platform.lower()}.com")

    # --- EXPERIENCE & QUALIFICATIONS ---
    st.markdown("---")
    st.subheader("Experience & Qualifications")
    st.write(
        """
    - ✔️ Describe your experience and qualifications here.
    """
    )

    # --- SKILLS ---
    st.markdown("---")
    st.subheader("Hard Skills")
    st.write(
        """
    - Describe your hard skills here.
    """
    )

    # --- WORK HISTORY ---
    st.markdown("---")
    st.subheader("Work History")
    st.write("---")

    # --- JOB 1 ---
    st.write("🚧", "**Job Title 1 | Company Name**")
    start_date_job1 = st.date_input("Start Date (Job 1)", value=None, key="start_date_job1")
    end_date_job1 = st.date_input("End Date (Job 1)", value=None, key="end_date_job1")
    responsibilities_job1 = st.text_area("Responsibilities (Job 1)", "")

    # --- JOB 2 ---
    st.write('\n')
    st.write("🚧", "**Job Title 2 | Company Name**")
    start_date_job2 = st.date_input("Start Date (Job 2)", value=None, key="start_date_job2")
    end_date_job2 = st.date_input("End Date (Job 2)", value=None, key="end_date_job2")
    responsibilities_job2 = st.text_area("Responsibilities (Job 2)", "")

    # --- JOB 3 ---
    st.write('\n')
    st.write("🚧", "**Job Title 3 | Company Name**")
    start_date_job3 = st.date_input("Start Date (Job 3)", value=None, key="start_date_job3")
    end_date_job3 = st.date_input("End Date (Job 3)", value=None, key="end_date_job3")
    responsibilities_job3 = st.text_area("Responsibilities (Job 3)", "")

    # --- Projects & Accomplishments ---
    st.markdown("---")
    st.subheader("Projects & Accomplishments")
    st.write("---")
    project_name = st.text_input("Project Name", "")
    project_link = st.text_input("Project Link", "")

    # --- Preview and Download Resume Buttons ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    # if col1.button("Preview Resume", key="preview_resume"):
    #     buffer = BytesIO()
    #     doc = SimpleDocTemplate(buffer, pagesize=letter)
    #     styles = getSampleStyleSheet()
    #     normal_style = styles['Normal']
    #     title_style = styles['Title']
    #     heading_style = styles['Heading1']
        
    #     elements = []
        
    #     elements.append(Paragraph(name, title_style))
    #     elements.append(Spacer(1, 12))
    #     elements.append(Paragraph(description, normal_style))
    #     elements.append(Spacer(1, 12))
        
    #     elements.append(Paragraph("Social Media:", heading_style))
    #     for platform, link in social_media.items():
    #         elements.append(Paragraph(f"{platform}: {link}", normal_style))
        
    #     elements.append(Spacer(1, 12))
    #     elements.append(Paragraph("Experience & Qualifications:", heading_style))
    #     elements.append(Paragraph("- ✔️ Describe your experience and qualifications here.", normal_style))
        
    #     elements.append(Spacer(1, 12))
    #     elements.append(Paragraph("Hard Skills:", heading_style))
    #     elements.append(Paragraph("- Describe your hard skills here.", normal_style))
        
    #     elements.append(Spacer(1, 12))
    #     elements.append(Paragraph("Work History:", heading_style))
    #     elements.append(Spacer(1, 6))
        
    #     # --- JOB 1 ---
    #     elements.append(Paragraph("🚧 Job Title 1 | Company Name", normal_style))
    #     elements.append(Paragraph(f"Start Date: {start_date_job1}  End Date: {end_date_job1}", normal_style))
    #     elements.append(Paragraph(f"Responsibilities: {responsibilities_job1}", normal_style))
        
    #     # --- JOB 2 ---
    #     elements.append(Paragraph("🚧 Job Title 2 | Company Name", normal_style))
    #     elements.append(Paragraph(f"Start Date: {start_date_job2}  End Date: {end_date_job2}", normal_style))
    #     elements.append(Paragraph(f"Responsibilities: {responsibilities_job2}", normal_style))
        
    #     # --- JOB 3 ---
    #     elements.append(Paragraph("🚧 Job Title 3 | Company Name", normal_style))
    #     elements.append(Paragraph(f"Start Date: {start_date_job3}  End Date: {end_date_job3}", normal_style))
    #     elements.append(Paragraph(f"Responsibilities: {responsibilities_job3}", normal_style))
        
    #     elements.append(Spacer(1, 12))
    #     elements.append(Paragraph("Projects & Accomplishments:", heading_style))
    #     elements.append(Paragraph(f"{project_name}: {project_link}", normal_style))
        
    #     doc.build(elements)
    #     buffer.seek(0)
    #     st.write(buffer.getvalue())
    if col2.button("Download Resume", key="download_resume"):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        normal_style = styles['Normal']
        title_style = styles['Title']
        heading_style = styles['Heading1']
        
        elements = []
        
        elements.append(Paragraph(name, title_style))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(description, normal_style))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Social Media:", heading_style))
        for platform, link in social_media.items():
            elements.append(Paragraph(f"{platform}: {link}", normal_style))
        
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Experience & Qualifications:", heading_style))
        elements.append(Paragraph("- ✔️ Describe your experience and qualifications here.", normal_style))
        
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Hard Skills:", heading_style))
        elements.append(Paragraph("- Describe your hard skills here.", normal_style))
        
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Work History:", heading_style))
        elements.append(Spacer(1, 6))
        
        # --- JOB 1 ---
        elements.append(Paragraph("🚧 Job Title 1 | Company Name", normal_style))
        elements.append(Paragraph(f"Start Date: {start_date_job1}  End Date: {end_date_job1}", normal_style))
        elements.append(Paragraph(f"Responsibilities: {responsibilities_job1}", normal_style))
        
        # --- JOB 2 ---
        elements.append(Paragraph("🚧 Job Title 2 | Company Name", normal_style))
        elements.append(Paragraph(f"Start Date: {start_date_job2}  End Date: {end_date_job2}", normal_style))
        elements.append(Paragraph(f"Responsibilities: {responsibilities_job2}", normal_style))
        
        # --- JOB 3 ---
        elements.append(Paragraph("🚧 Job Title 3 | Company Name", normal_style))
        elements.append(Paragraph(f"Start Date: {start_date_job3}  End Date: {end_date_job3}", normal_style))
        elements.append(Paragraph(f"Responsibilities: {responsibilities_job3}", normal_style))
        
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Projects & Accomplishments:", heading_style))
        elements.append(Paragraph(f"{project_name}: {project_link}", normal_style))
        
        doc.build(elements)
        buffer.seek(0)
        st.download_button(
            label="📄 Download",
            data=buffer,
            file_name="CV.pdf",
            mime="application/pdf"
        )
