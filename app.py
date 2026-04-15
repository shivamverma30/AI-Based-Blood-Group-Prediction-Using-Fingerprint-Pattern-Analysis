import streamlit as st
import numpy as np
import os
import re
import smtplib
import tempfile
import base64
from email.message import EmailMessage
from datetime import datetime
from io import BytesIO

TF_AVAILABLE = True
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
except ImportError:
    TF_AVAILABLE = False
    load_model = None
    keras_image = None
    resnet_preprocess = None
    vgg_preprocess = None

from PIL import Image
import torch
import torch.nn.functional as torch_f
from torchvision.models import swin_t
from torchvision import transforms as torch_transforms
import requests

HF_HUB_AVAILABLE = True
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    HF_HUB_AVAILABLE = False
    hf_hub_download = None

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


# -------------------------------------------------
# Load Email Credentials
# -------------------------------------------------
EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="HemoScan AI | Blood Group Detection",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# -------------------------------------------------
# Custom CSS for Professional Medical Theme
# -------------------------------------------------
def load_custom_css():
    st.markdown("""
    <style>
        /* Main Theme Colors */
        :root {
            --primary-red: #C41E3A;
            --dark-red: #8B0000;
            --soft-grey: #F5F5F5;
            --medical-white: #FAFAFA;
            --text-dark: #2C3E50;
            --accent-gold: #D4AF37;
        }
        
        /* Global Styles */
        .main {
            background: linear-gradient(135deg, #FAFAFA 0%, #F0F0F0 100%);
        }
        
        /* Header Styling */
        .medical-header {
            background: linear-gradient(135deg, #C41E3A 0%, #8B0000 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(196, 30, 58, 0.3);
        }
        
        .medical-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: 1px;
        }
        
        .medical-header p {
            font-size: 1.1rem;
            opacity: 0.9;
            margin-top: 0.5rem;
        }
        
        /* Card Styling */
        .medical-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            border-left: 4px solid #C41E3A;
            margin-bottom: 1rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .medical-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Model Selection Card */
        .model-selection-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        /* Button Styling */
        .stButton>button {
            background: linear-gradient(135deg, #C41E3A 0%, #8B0000 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(196, 30, 58, 0.4);
        }
        
        .stButton>button:active {
            transform: translateY(0);
        }
        
        /* Secondary Button */
        .secondary-btn>button {
            background: linear-gradient(135deg, #34495E 0%, #2C3E50 100%);
        }
        
        /* File Uploader Styling */
        .stFileUploader {
            background: linear-gradient(145deg, #1c2331, #0f172a);
            border: 2px dashed #C41E3A;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .stFileUploader:hover {
            border-color: #ff4d4d;
            background: linear-gradient(145deg, #20293b, #111827);
            box-shadow: 0 0 15px rgba(196, 30, 58, 0.4);
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            padding: 1.5rem;
        }

        [data-testid="stFileUploaderFile"] {
            background: #111827;
            border-radius: 8px;
            padding: 8px;
        }
        
        /* Input Fields */
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            border-radius: 8px;
            border: 2px solid #E0E0E0;
            padding: 0.75rem;
            transition: border-color 0.3s ease;
        }
        
        .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
            border-color: #C41E3A;
            box-shadow: 0 0 0 3px rgba(196, 30, 58, 0.1);
        }
        
        /* Select Box */
        .stSelectbox>div>div>div {
            border-radius: 8px;
            border: 2px solid #E0E0E0;
        }
        
        /* Success/Error Messages */
        .stSuccess {
            background: linear-gradient(135deg, #D4EDDA 0%, #C3E6CB 100%);
            border-left: 4px solid #28A745;
            border-radius: 8px;
            padding: 1rem;
        }
        
        .stError {
            background: linear-gradient(135deg, #F8D7DA 0%, #F5C6CB 100%);
            border-left: 4px solid #DC3545;
            border-radius: 8px;
            padding: 1rem;
        }
        
        /* Progress Bar */
        .stProgress>div>div>div {
            background: linear-gradient(90deg, #C41E3A 0%, #8B0000 100%);
            border-radius: 10px;
        }
        
        /* Result Box */
        .result-box {
            background: linear-gradient(135deg, #C41E3A 0%, #8B0000 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(196, 30, 58, 0.3);
            margin: 1rem 0;
        }
        
        .result-box h2 {
            margin: 0;
            font-size: 3rem;
            font-weight: 700;
        }
        
        .result-box p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        /* Confidence Indicator */
        .confidence-high { color: #28A745; font-weight: bold; }
        .confidence-medium { color: #FFC107; font-weight: bold; }
        .confidence-low { color: #DC3545; font-weight: bold; }
        
        /* Footer */
        .medical-footer {
            text-align: center;
            padding: 2rem;
            color: #7F8C8D;
            font-size: 0.9rem;
            margin-top: 3rem;
            border-top: 1px solid #E0E0E0;
        }
        
        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-in {
            animation: fadeIn 0.6s ease-out;
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


load_custom_css()


# -------------------------------------------------
# Email Validation
# -------------------------------------------------
def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email)


# -------------------------------------------------
# Prediction Function (Enhanced with Error Handling)
# -------------------------------------------------
def predict_blood_group(model, uploaded_file, input_size, class_labels):
    try:
        # Keras/TensorFlow path (existing behavior)
        if TF_AVAILABLE and hasattr(model, "predict"):
            img = keras_image.load_img(uploaded_file, target_size=input_size)
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            #IMPORTANT CHANGE
            if "resnet" in str(type(model)).lower():
                img_array = resnet_preprocess(img_array)
            else:
                img_array = vgg_preprocess(img_array)
            predictions = model.predict(img_array, verbose=0)
            pred_index = np.argmax(predictions)
            confidence = predictions[0][pred_index] * 100
        else:
            # PyTorch path (.pth Swin Transformer) — EXACT same as training notebook
            uploaded_file.seek(0)
            img = Image.open(uploaded_file).convert("RGB")
            transform = torch_transforms.Compose([
                torch_transforms.Resize((224, 224)),
                torch_transforms.ToTensor(),
                torch_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            img_tensor = transform(img)
            xb = img_tensor.unsqueeze(0)
            model.eval()
            with torch.no_grad():
                outputs = model(xb)
                probs = torch.softmax(outputs, dim=1)
            pred_index = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_index].item() * 100

        predicted_label = class_labels[pred_index]

        return predicted_label, confidence, None
    except Exception as e:
        return None, None, str(e)


# -------------------------------------------------
# Enhanced PDF Generator - Single Page Optimized
# -------------------------------------------------
def generate_pdf(user_data, predicted_label, confidence, uploaded_file):
    try:
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(
            temp_pdf.name,
            pagesize=letter,
            rightMargin=42,
            leftMargin=42,
            topMargin=30,
            bottomMargin=24
        )

        elements = []

        report_timestamp = datetime.now()
        report_id = f"HS-{report_timestamp.strftime('%Y%m%d%H%M%S')}"

        # Typography system
        styles = getSampleStyleSheet()

        header_title_style = ParagraphStyle(
            'HeaderTitle',
            parent=styles['Heading1'],
            fontSize=21,
            textColor=colors.HexColor('#C41E3A'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=24,
            spaceAfter=2
        )

        header_subtitle_style = ParagraphStyle(
            'HeaderSubtitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#5E6A71'),
            alignment=TA_CENTER,
            fontName='Helvetica',
            leading=12,
            spaceAfter=8
        )

        section_title_style = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontSize=11.5,
            textColor=colors.HexColor('#2C3E50'),
            fontName='Helvetica-Bold',
            spaceAfter=6,
            leading=13
        )

        body_label_style = ParagraphStyle(
            'BodyLabel',
            parent=styles['Normal'],
            fontSize=8.7,
            textColor=colors.HexColor('#2F3E46'),
            fontName='Helvetica-Bold',
            leading=11,
            spaceAfter=0
        )

        body_value_style = ParagraphStyle(
            'BodyValue',
            parent=styles['Normal'],
            fontSize=8.7,
            textColor=colors.HexColor('#2C3E50'),
            fontName='Helvetica',
            leading=11,
            spaceAfter=0
        )

        result_title_style = ParagraphStyle(
            'ResultTitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#6C757D'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=12,
            spaceAfter=3
        )

        result_value_style = ParagraphStyle(
            'ResultValue',
            parent=styles['Normal'],
            fontSize=31,
            textColor=colors.HexColor('#8B0000'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=32,
            spaceAfter=2
        )

        result_conf_style = ParagraphStyle(
            'ResultConf',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=13,
            spaceAfter=0
        )

        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=7.1,
            textColor=colors.HexColor('#606D75'),
            alignment=TA_CENTER,
            fontName='Helvetica-Oblique',
            leading=9,
            spaceAfter=4
        )

        signature_style = ParagraphStyle(
            'Signature',
            parent=styles['Normal'],
            fontSize=7.8,
            textColor=colors.HexColor('#2C3E50'),
            alignment=TA_CENTER,
            spaceAfter=1,
            leading=9
        )

        # ---------- Header ----------
        elements.append(Paragraph("<font color='#C41E3A' size='20'>✚</font>", ParagraphStyle(
            'Logo', parent=styles['Normal'], alignment=TA_CENTER, leading=16, spaceAfter=2
        )))
        elements.append(Paragraph("HEMOSCAN AI", header_title_style))
        elements.append(Paragraph("Blood Group Detection Report", header_subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.3, color=colors.HexColor('#C41E3A'), spaceAfter=8))

        # ---------- Patient Information ----------
        elements.append(Paragraph("Patient Information", section_title_style))

        table_data = [
            [Paragraph("Full Name", body_label_style), Paragraph(str(user_data["name"]), body_value_style)],
            [Paragraph("Age", body_label_style), Paragraph(str(user_data["age"]), body_value_style)],
            [Paragraph("Gender", body_label_style), Paragraph(str(user_data["gender"]), body_value_style)],
            [Paragraph("Phone", body_label_style), Paragraph(str(user_data["phone"]), body_value_style)],
            [Paragraph("Email", body_label_style), Paragraph(str(user_data["email"]), body_value_style)],
            [Paragraph("Report Date", body_label_style), Paragraph(report_timestamp.strftime("%Y-%m-%d %H:%M:%S"), body_value_style)],
            [Paragraph("Report ID", body_label_style), Paragraph(report_id, body_value_style)]
        ]

        patient_table = Table(table_data, colWidths=[1.55*inch, 4.2*inch])
        patient_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor('#FBFCFD')),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor('#F5F7F9'), colors.white]),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor('#2C3E50')),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("INNERGRID", (0, 0), (-1, -1), 0.45, colors.HexColor('#D7DEE4')),
            ("BOX", (0, 0), (-1, -1), 1.1, colors.HexColor('#C41E3A')),
        ]))

        elements.append(patient_table)
        elements.append(Spacer(1, 0.12*inch))

        # ---------- Result Section ----------
        elements.append(Paragraph("AI Prediction Result", section_title_style))

        if confidence >= 90:
            conf_color = colors.HexColor('#1B8A3C')
            conf_text = "High Confidence"
        elif confidence >= 70:
            conf_color = colors.HexColor('#D9822B')
            conf_text = "Medium Confidence"
        else:
            conf_color = colors.HexColor('#B42318')
            conf_text = "Low Confidence"

        confidence_para = Paragraph(
            f"<font color='{conf_color.hexval()}'>Confidence: {confidence:.2f}% ({conf_text})</font>",
            result_conf_style
        )

        result_table = Table([
            [Paragraph("Predicted Blood Group", result_title_style)],
            [Paragraph(f"{predicted_label}", result_value_style)],
            [confidence_para],
        ], colWidths=[5.75*inch], rowHeights=[0.30*inch, 0.56*inch, 0.30*inch])
        result_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor('#FFF3F3')),
            ("BOX", (0, 0), (-1, -1), 1.4, colors.HexColor('#C41E3A')),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))

        elements.append(result_table)
        elements.append(Spacer(1, 0.12*inch))

        # ---------- Fingerprint Image ----------
        elements.append(Paragraph("Fingerprint Sample", section_title_style))

        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img = Image.open(uploaded_file)
        img.thumbnail((175, 175), Image.Resampling.LANCZOS)
        img.save(temp_img.name, quality=90)

        img_table = Table(
            [[RLImage(temp_img.name, width=1.35*inch, height=1.35*inch)]],
            colWidths=[5.75*inch],
            rowHeights=[1.55*inch]
        )
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor('#CFD6DD')),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor('#F9FAFB')),
        ]))
        elements.append(img_table)
        elements.append(Spacer(1, 0.13*inch))

        # ---------- Footer ----------
        elements.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor('#D9E0E6'), spaceBefore=2, spaceAfter=6))

        disclaimer_text = """
        <b>Disclaimer:</b> This report is generated by an AI-based system for informational purposes only. 
        It should not be used as a substitute for professional medical diagnosis or laboratory testing. 
        Always consult with qualified healthcare professionals for medical decisions.
        """
        elements.append(Paragraph(disclaimer_text, disclaimer_style))

        elements.append(HRFlowable(width="36%", thickness=0.5, color=colors.HexColor('#2C3E50'),
                                   spaceBefore=4, spaceAfter=3, hAlign='CENTER'))
        elements.append(Paragraph("Made with ❤️ by Shivam Verma and Anshu Raj", signature_style))
        elements.append(Paragraph(
            "© 2026 HemoScan AI",
            ParagraphStyle('Copyright', parent=signature_style, textColor=colors.HexColor('#95A5A6'), fontSize=7)
        ))

        # Build PDF
        doc.build(elements)

        return temp_pdf.name

    except Exception as e:
        st.error(f"PDF Generation Error: {str(e)}")
        return None


# -------------------------------------------------
# Enhanced Email Function with Better Error Handling
# -------------------------------------------------
def send_email(receiver_email, pdf_path, user_name):
    try:
        msg = EmailMessage()
        msg["Subject"] = f"HemoScan AI - Blood Group Report for {user_name}"
        msg["From"] = f"HemoScan AI <{EMAIL_ADDRESS}>"
        msg["To"] = receiver_email

        # HTML Email Body
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background: linear-gradient(135deg, #C41E3A 0%, #8B0000 100%); 
                           color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
                    <h1 style="margin: 0;">🩸 HemoScan AI</h1>
                    <p style="margin: 10px 0 0 0; opacity: 0.9;">Your Blood Group Analysis Report</p>
                </div>
                
                <div style="background: #f9f9f9; padding: 30px; border: 1px solid #ddd; 
                           border-top: none; border-radius: 0 0 10px 10px;">
                    <p>Dear <strong>{user_name}</strong>,</p>
                    
                    <p>Thank you for using HemoScan AI. Your blood group analysis report has been 
                    generated successfully and is attached to this email.</p>
                    
                    <div style="background: white; padding: 20px; border-left: 4px solid #C41E3A; 
                               margin: 20px 0; border-radius: 5px;">
                        <p style="margin: 0; color: #666; font-size: 14px;">
                           <strong>Important:</strong> This report is generated by an AI-based system 
                           for informational purposes only and should not replace professional medical testing.
                        </p>
                    </div>
                    
                    <p style="color: #666; font-size: 12px; margin-top: 30px;">
                        Best regards,<br>
                        <strong>HemoScan AI Team</strong><br>
                        Shivam Verma & Anshu Raj<br>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        msg.add_alternative(html_content, subtype='html')
        
        # Attach PDF
        with open(pdf_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="application",
                subtype="pdf",
                filename=f"HemoScan_Report_{user_name.replace(' ', '_')}.pdf"
            )

        # Send Email with better error handling
        try:
            server = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10)
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            return True, None
        except smtplib.SMTPAuthenticationError as e:
            return False, "Email authentication failed. Please check your email credentials in secrets."
        except smtplib.SMTPRecipientsRefused as e:
            return False, "Recipient email address was refused by the server."
        except smtplib.SMTPSenderRefused as e:
            return False, "Sender email address was refused by the server."
        except smtplib.SMTPException as e:
            return False, f"SMTP Error: {str(e)}"
        except Exception as e:
            return False, f"Connection Error: Unable to connect to email server. Please check your internet connection."
            
    except Exception as e:
        return False, f"Email preparation error: {str(e)}"


# -------------------------------------------------
# Model Selection Logic - Supports Multiple Models
# -------------------------------------------------
MODEL_FOLDER = "models"
SWIN_MODEL_FILE = "swin.pth"
SWIN_MODEL_URL = "https://huggingface.co/shivamverma30/hemoscan-model/resolve/main/swin.pth"
SWIN_MODEL_PATH = os.path.join(MODEL_FOLDER, SWIN_MODEL_FILE)


def download_model_from_hf():
    """Download the Swin Transformer checkpoint from Hugging Face safely."""
    if os.path.exists(SWIN_MODEL_PATH):
        try:
            torch.load(SWIN_MODEL_PATH, map_location="cpu", weights_only=False)
            return True
        except Exception:
            if os.path.exists(SWIN_MODEL_PATH):
                os.remove(SWIN_MODEL_PATH)

    try:
        with st.spinner("⏳ Downloading Swin Transformer model from Hugging Face..."):
            repo_id = "shivamverma30/hemoscan-model"
            filename = "swin.pth"

            downloaded_path = None
            download_errors = []

            if HF_HUB_AVAILABLE:
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=MODEL_FOLDER,
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                except Exception as e:
                    download_errors.append(str(e))

            if downloaded_path is None:
                download_url = SWIN_MODEL_URL if "download=true" in SWIN_MODEL_URL else f"{SWIN_MODEL_URL}?download=true"
                temp_file = tempfile.NamedTemporaryFile(delete=False, dir=MODEL_FOLDER, suffix=".tmp")
                downloaded_path = temp_file.name
                temp_file.close()

                try:
                    response = requests.get(download_url, stream=True, timeout=120, allow_redirects=True)
                    response.raise_for_status()

                    content_type = response.headers.get("content-type", "").lower()
                    if "html" in content_type:
                        raise ValueError("Hugging Face returned HTML instead of a model file.")

                    with open(downloaded_path, "wb") as model_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                model_file.write(chunk)
                except Exception as e:
                    download_errors.append(str(e))
                    if os.path.exists(downloaded_path):
                        os.remove(downloaded_path)
                    raise ValueError("; ".join(download_errors))

            try:
                torch.load(downloaded_path, map_location="cpu", weights_only=False)
            except Exception as e:
                if os.path.exists(downloaded_path):
                    os.remove(downloaded_path)
                raise ValueError(f"Downloaded file is corrupted: {str(e)}")

            if downloaded_path != SWIN_MODEL_PATH:
                os.replace(downloaded_path, SWIN_MODEL_PATH)

            st.success("✅ Model downloaded and validated successfully!")
            return True

    except Exception as e:
        st.error(f"❌ Failed to download Swin model: {str(e)}")
        st.info("💡 Troubleshooting: Check your internet connection and verify the HF repo is accessible.")
        return False

# Create models folder if it doesn't exist
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Get all supported model files
models_list = [
    f for f in os.listdir(MODEL_FOLDER)
    if f.endswith(".pth") or (TF_AVAILABLE and f.endswith(".keras"))
]

# Cloud fallback: download Swin model when no local model is available.
if len(models_list) == 0:
    download_model_from_hf()
    models_list = [
        f for f in os.listdir(MODEL_FOLDER)
        if f.endswith(".pth") or (TF_AVAILABLE and f.endswith(".keras"))
    ]

if len(models_list) == 0:
    st.warning("⚠️ No models found in /models folder yet. The UI will still load, and the app will try Hugging Face when available.")
    models_list = [SWIN_MODEL_FILE]

# Model configurations - Add your models here
AVAILABLE_MODELS = {
    "default": {
        "name": "Auto-Detect",
        "input_size": None,
        "classes": ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
    }
}

# Add detected models to available models
for model_file in models_list:
    if model_file not in AVAILABLE_MODELS:
        AVAILABLE_MODELS[model_file] = {
            "name": model_file.replace(".keras", "").replace(".pth", "").replace("_", " ").title(),
            "input_size": None,
            "classes": ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        }


# -------------------------------------------------
# Main Content Area
# -------------------------------------------------
st.markdown("""
<div class="medical-header animate-in">
    <h1>🩸 Blood Group Detection System</h1>
    <p>AI-Based Blood Group Prediction Using Fingerprint Pattern Analysis</p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# MODEL SELECTION - AT TOP (Above Patient Information)
# -------------------------------------------------
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 1.5rem; border-radius: 12px; 
            margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
    <h3 style="margin: 0 0 1rem 0; color: white;">🤖 Model Selection</h3>
</div>
""", unsafe_allow_html=True)

model_col1, model_col2 = st.columns([2, 1])

with model_col1:
    # Model Selection Dropdown
    selected_model_file = st.selectbox(
        "Choose AI Model for Analysis", 
        models_list,
        format_func=lambda x: AVAILABLE_MODELS.get(x, {}).get("name", x),
        help="Select the trained model for blood group prediction"
    )

# Load Model with Caching - Dynamic loading based on selection
@st.cache_resource(show_spinner=False)
def load_model_cached(model_path):
    try:
        if model_path.endswith(".keras"):
            if not TF_AVAILABLE:
                st.error("TensorFlow is not installed. Please select a PyTorch (.pth) model.")
                return None
            return load_model(model_path, compile=False)

        if model_path.endswith(".pth"):
            # Pre-validation: Check if file exists and is readable
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                return None

            # Pre-validation: Check if it's a valid ZIP/PyTorch file
            try:
                with open(model_path, "rb") as f:
                    header = f.read(4)
                    if header[:2] != b'PK':  # ZIP file magic number (0x504B)
                        st.error("❌ Model file is corrupted (invalid ZIP header). Attempting to redownload...")
                        if os.path.exists(model_path):
                            os.remove(model_path)
                        if download_model_from_hf():
                            return load_model_cached(model_path)  # Recursive retry after redownload
                        return None
            except Exception as e:
                st.error(f"Cannot read model file: {str(e)}")
                return None

            default_classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
            num_classes = len(default_classes)

            # Try to load with weights_only=False for compatibility
            try:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            except Exception as e:
                st.error(f"❌ Failed to load checkpoint: {str(e)}")
                st.info("💡 The model file appears corrupted. Attempting to redownload...")
                if os.path.exists(model_path):
                    os.remove(model_path)
                if download_model_from_hf():
                    return load_model_cached(model_path)  # Recursive retry
                return None

            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Remove common wrapper prefixes if present.
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if new_key.startswith("module."):
                    new_key = new_key[len("module."):]
                if new_key.startswith("model."):
                    new_key = new_key[len("model."):]
                cleaned_state_dict[new_key] = v

            # Infer number of classes from final head if available.
            if "head.weight" in cleaned_state_dict and len(cleaned_state_dict["head.weight"].shape) == 2:
                num_classes = int(cleaned_state_dict["head.weight"].shape[0])

            class NetWrapper(torch.nn.Module):
                def __init__(self, backbone):
                    super().__init__()
                    self.backbone = backbone

                def forward(self, x):
                    return self.backbone(x)

            backbone = swin_t(weights=None)
            in_features = backbone.head.in_features
            backbone.head = torch.nn.Linear(in_features, num_classes)

            torch_model = NetWrapper(backbone)

            torch_model.load_state_dict(cleaned_state_dict, strict=False)
            torch_model.eval()

            return torch_model

        raise ValueError("Unsupported model file extension")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model_path = os.path.join(MODEL_FOLDER, selected_model_file)
model = None
default_classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
input_size = (224, 224)
class_labels = default_classes

with model_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.success(f" Loaded")

# Display model info below selection
info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    st.info(f"**Model:** {selected_model_file}")
with info_col2:
    st.info(f"**Input Size:** {input_size[0]}x{input_size[1]}")
with info_col3:
    st.info(f"**Classes:** {len(class_labels)} ({', '.join(class_labels)})")

st.markdown("---")


# -------------------------------------------------
# Session State Management
# -------------------------------------------------
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
    st.session_state.prediction_data = None
    st.session_state.user_data = None
    st.session_state.email_status = None
    st.session_state.uploaded_file_bytes = None
    st.session_state.uploaded_file_name = None
    st.session_state.uploaded_file_type = None

def persist_uploaded_file():
    uploaded = st.session_state.get("fingerprint_upload")
    if uploaded is None:
        st.session_state.uploaded_file_bytes = None
        st.session_state.uploaded_file_name = None
        st.session_state.uploaded_file_type = None
        return

    st.session_state.uploaded_file_bytes = uploaded.getvalue()
    st.session_state.uploaded_file_name = uploaded.name
    st.session_state.uploaded_file_type = uploaded.type

def get_uploaded_file_stream():
    file_bytes = st.session_state.get("uploaded_file_bytes")
    if not file_bytes:
        return None

    uploaded_stream = BytesIO(file_bytes)
    uploaded_stream.name = st.session_state.get("uploaded_file_name") or "uploaded_image"
    return uploaded_stream

# -------------------------------------------------
# Dashboard Layout - Patient Information
# -------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="medical-card animate-in">
        <h3 style="color: #C41E3A; margin-top: 0;">👤 Patient Information</h3>
    </div>
    """, unsafe_allow_html=True)
    
    name = st.text_input("Full Name*", placeholder="Enter patient full name")

    col_age, col_gender = st.columns(2)
    with col_age:
        age = st.number_input("Age*", min_value=1, max_value=120, value=25)
    with col_gender:
        gender = st.selectbox("Gender*", ["Male", "Female", "Other"])

    phone = st.text_input("Phone Number*", placeholder="(+91) 98765 43210")
    email = st.text_input("Email Address*", placeholder="patient@gmail.com")

    st.markdown("---")

    st.file_uploader(
        "📤 Upload Fingerprint Image (Max 5MB)",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear fingerprint image for analysis (Max size: 5MB)",
        key="fingerprint_upload",
        on_change=persist_uploaded_file
    )

    submitted = st.button("🔬 Analyze Sample", use_container_width=True)


# Handle Form Submission
if submitted:
    uploaded_file = get_uploaded_file_stream()

    # Validation
    validation_errors = []

    if not name.strip():
        validation_errors.append("Please enter patient name")
    if not validate_email(email):
        validation_errors.append("Please enter a valid email address")
    if not phone.strip():
        validation_errors.append("Please enter phone number")
    if uploaded_file is None:
        validation_errors.append("Please upload a fingerprint image")
    
    if validation_errors:
        for error in validation_errors:
            st.error(f" {error}")
        st.stop()
    
    if model is None:
        model = load_model_cached(model_path)
    
    if model is None:
        st.info("Model not ready yet. The interface is available, but prediction will be disabled until a valid model is loaded.")
    else:
        # Clear previous email status on new submission
        st.session_state.email_status = None
        
        # Process Prediction
        with st.spinner(" Analyzing fingerprint pattern..."):
            predicted_label, confidence, error = predict_blood_group(
                model, uploaded_file, input_size, class_labels
            )
        
        if error:
            st.error(f" Analysis Error: {error}")
            st.stop()
        
        # Store Data
        user_data = {
            "name": name,
            "age": age,
            "gender": gender,
            "phone": phone,
            "email": email
        }
        
        st.session_state.user_data = user_data
        st.session_state.prediction_data = {
            "label": predicted_label,
            "confidence": confidence,
            "file": uploaded_file
        }


# Display Results in Right Column
with col2:
    if st.session_state.prediction_data:
        pred_data = st.session_state.prediction_data
        user_data = st.session_state.user_data
        
        st.markdown("""
        <div class="medical-card animate-in">
            <h3 style="color: #C41E3A; margin-top: 0;">📊 Analysis Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Result Display
        st.markdown(f"""
        <div class="result-box animate-in">
            <h2>{pred_data['label']}</h2>
            <p>Predicted Blood Group</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence with Color Coding
        conf = pred_data['confidence']
        if conf >= 90:
            conf_class = "confidence-high"
            conf_text = "High Confidence"
        elif conf >= 70:
            conf_class = "confidence-medium"
            conf_text = "Medium Confidence"
        else:
            conf_class = "confidence-low"
            conf_text = "Low Confidence - Verification Recommended"
        
        st.markdown(f"""
        <div style="text-align: center; margin: 1rem 0;">
            <p style="font-size: 1.2rem; margin: 0;">
                Confidence: <span class="{conf_class}">{conf:.2f}%</span>
            </p>
            <p style="color: #7F8C8D; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                {conf_text}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress Bar
        st.progress(int(conf))
        
        # Image Preview
        st.markdown("### 🔍 Sample Preview")
        pred_data["file"].seek(0)
        img = Image.open(pred_data['file'])
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(img, width=200, caption="Uploaded Fingerprint")
        
        # Generate PDF
        if st.session_state.pdf_path is None:
            with st.spinner("📄 Generating medical report..."):
                pred_data["file"].seek(0)
                pdf_path = generate_pdf(user_data, pred_data['label'], 
                                       pred_data['confidence'], pred_data['file'])
                st.session_state.pdf_path = pdf_path
        
        # Action Buttons
        if st.session_state.pdf_path:
            col_download, col_email = st.columns(2)
            
            with col_download:
                with open(st.session_state.pdf_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Report",
                        data=f,
                        file_name=f"HemoScan_Report_{user_data['name'].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            
            with col_email:
                if st.button("📧 Email Report", use_container_width=True):
                    with st.spinner("Sending email..."):
                        success, error_msg = send_email(
                            user_data['email'], 
                            st.session_state.pdf_path,
                            user_data['name']
                        )
                        
                        if success:
                            st.session_state.email_status = ("success", "✅ Report sent successfully!")
                        else:
                            st.session_state.email_status = ("error", f"❌ {error_msg}")
        
        # Display email status message below buttons (no animation)
        if st.session_state.email_status:
            status_type, status_msg = st.session_state.email_status
            if status_type == "success":
                st.success(status_msg)
            else:
                st.error(status_msg)
        
        # Reset Button
        if st.button("🔄 New Analysis", type="secondary"):
            st.session_state.pdf_path = None
            st.session_state.prediction_data = None
            st.session_state.user_data = None
            st.session_state.email_status = None
            st.rerun()


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("""
<div class="medical-footer">
    <p><strong>HemoScan AI</strong> | Advanced Blood Group Detection System</p>
    <p style="font-size: 0.8rem; color: #95A5A6;">
        This application uses machine learning for educational and research purposes only.
        Not intended for clinical diagnosis without professional verification.
    </p>
</div>
""", unsafe_allow_html=True)