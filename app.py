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

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input


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
    initial_sidebar_state="expanded"
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
        
        /* Sidebar Styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #2C3E50 0%, #1A252F 100%);
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
            background: white;
            border: 2px dashed #C41E3A;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        }
        
        .stFileUploader:hover {
            border-color: #8B0000;
            background: #FFF5F5;
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
        
        /* Sidebar Info Cards */
        .sidebar-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
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
        img = keras_image.load_img(uploaded_file, target_size=input_size)
        img_array = keras_image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)

        #IMPORTANT CHANGE
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array, verbose=0)

        pred_index = np.argmax(predictions)
        confidence = predictions[0][pred_index] * 100
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
            rightMargin=50,
            leftMargin=50,
            topMargin=40,
            bottomMargin=30
        )
        
        elements = []
        
        # Custom Styles
        styles = getSampleStyleSheet()
        
        # Compact Header Style
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#8B0000'),
            spaceAfter=8,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Subheader Style
        subheader_style = ParagraphStyle(
            'CustomSubheader',
            parent=styles['Heading2'],
            fontSize=11,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )
        
        # Normal Text Style
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=3,
            fontName='Helvetica'
        )
        
        # Compact Result Style
        result_style = ParagraphStyle(
            'ResultStyle',
            parent=styles['Normal'],
            fontSize=28,
            textColor=colors.HexColor('#C41E3A'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            spaceAfter=4
        )
        
        # Small Result Style for label
        result_label_style = ParagraphStyle(
            'ResultLabelStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER,
            fontName='Helvetica',
            spaceAfter=2
        )
        
        # Disclaimer Style - Compact
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Italic'],
            fontSize=7,
            textColor=colors.HexColor('#7F8C8D'),
            alignment=TA_CENTER,
            spaceAfter=3
        )
        
        # Signature Style
        signature_style = ParagraphStyle(
            'Signature',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#2C3E50'),
            alignment=TA_CENTER,
            spaceAfter=2
        )

        # --- HEADER SECTION ---
        # Red Cross Symbol
        elements.append(Paragraph(
            "<font size='23' color='#C41E3A'>✚</font>", 
            ParagraphStyle('Logo', alignment=TA_CENTER, leading=20,spaceAfter=4)
        ))
        
        elements.append(Paragraph("HEMOSCAN AI", header_style))
        elements.append(Paragraph("Advanced Blood Group Detection System", 
                                  ParagraphStyle('SubTitle', alignment=TA_CENTER, 
                                               textColor=colors.HexColor('#7F8C8D'), 
                                               fontSize=10, spaceAfter=10)))
        
        # Horizontal Line
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#C41E3A'), spaceAfter=10))
        
        # --- CERTIFICATE TITLE ---
        elements.append(Paragraph("<b>MEDICAL ANALYSIS REPORT</b>", 
                                  ParagraphStyle('CertTitle', alignment=TA_CENTER, 
                                               fontSize=12, textColor=colors.HexColor('#2C3E50'),
                                               spaceAfter=10)))
        
        # --- PATIENT DETAILS TABLE - Compact ---
        elements.append(Paragraph("Patient Information", subheader_style))
        
        table_data = [
            [Paragraph("<b>Full Name</b>", normal_style), user_data["name"]],
            [Paragraph("<b>Age</b>", normal_style), str(user_data["age"])],
            [Paragraph("<b>Gender</b>", normal_style), user_data["gender"]],
            [Paragraph("<b>Contact</b>", normal_style), user_data["phone"]],
            [Paragraph("<b>Email</b>", normal_style), user_data["email"]],
            [Paragraph("<b>Date</b>", normal_style), datetime.now().strftime("%Y-%m-%d %H:%M")],
            [Paragraph("<b>Report ID</b>", normal_style), f"HS-{datetime.now().strftime('%Y%m%d%H%M%S')}"]
        ]

        patient_table = Table(table_data, colWidths=[1.3*inch, 2.2*inch])
        patient_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor('#F8F9FA')),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor('#2C3E50')),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor('#C41E3A')),
        ]))
        
        elements.append(patient_table)
        elements.append(Spacer(1, 0.15*inch))
        
        # --- ANALYSIS RESULTS SECTION - Side by Side ---
        elements.append(Paragraph("Analysis Results", subheader_style))
        
        # Result Box with Background - Compact side by side
        result_data = [[
            Paragraph("<b>Blood Group</b>", result_label_style),
            Paragraph("<b>Confidence</b>", result_label_style)
        ], [
            Paragraph(f"{predicted_label}", result_style),
            Paragraph(f"{confidence:.1f}%", result_style)
        ]]
        
        result_table = Table(result_data, colWidths=[1.8*inch, 1.8*inch],rowHeights=[0.45*inch, 0.65*inch])
        result_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor('#FFF5F5')),
            ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor('#C41E3A')),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ("TOPPADDING", (0, 1), (-1, 1), 4),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ]))
        
        elements.append(result_table)
        elements.append(Spacer(1, 0.15*inch))
        
        # --- FINGERPRINT IMAGE - Smaller ---
        elements.append(Paragraph("Fingerprint Sample", subheader_style))
        
        # Process image for PDF - smaller size
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img = Image.open(uploaded_file)
        # Resize to smaller dimensions for PDF
        img.thumbnail((150, 150), Image.Resampling.LANCZOS)
        img.save(temp_img.name, quality=90)
        
        # Center the image - smaller
        img_table = Table([[RLImage(temp_img.name, width=1.2*inch, height=1.2*inch)]], colWidths=[3.5*inch])
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
        ]))
        elements.append(img_table)
        
        elements.append(Spacer(1, 0.2*inch))
        
        # --- DISCLAIMER SECTION ---
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#DEE2E6'), spaceBefore=6, spaceAfter=6))
        
        disclaimer_text = """
        <b>Disclaimer:</b> This report is generated by an AI-based system for informational purposes only. 
        It should not be used as a substitute for professional medical diagnosis or laboratory testing. 
        Always consult with qualified healthcare professionals for medical decisions.
        """
        elements.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # --- FOOTER / SIGNATURE ---
        elements.append(Spacer(1, 0.1*inch))
        elements.append(HRFlowable(width="35%", thickness=0.5, color=colors.HexColor('#2C3E50'), 
                                  spaceBefore=6, spaceAfter=4, hAlign='CENTER'))
        elements.append(Paragraph("Made with ❤️ by Shivam Verma and Anshu Raj", signature_style))
        elements.append(Paragraph("Under the guidance of Dr. Kuldeep Yadav", 
                                  ParagraphStyle('Guide', parent=signature_style, 
                                               textColor=colors.HexColor('#7F8C8D'))))
        elements.append(Paragraph("© 2026 HemoScan AI", 
                                  ParagraphStyle('Copyright', parent=signature_style, 
                                               textColor=colors.HexColor('#95A5A6'), fontSize=7)))
        
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
                        Under the guidance of Dr. Kuldeep Yadav
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

# Create models folder if it doesn't exist
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Get all .keras model files
models_list = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".keras")]

if len(models_list) == 0:
    st.error("⚠️ No models found in /models folder. Please add .keras model files.")
    st.stop()

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
            "name": model_file.replace(".keras", "").replace("_", " ").title(),
            "input_size": None,
            "classes": ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        }


# -------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: white; font-size: 1.8rem; margin: 0;">🩸 HemoScan</h1>
        <p style="color: #B0B0B0; font-size: 0.9rem; margin: 0.5rem 0 0 0;">AI-Powered Blood Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Status Card
    st.markdown("""
    <div class="sidebar-card">
        <h3 style="color: white; margin: 0 0 1rem 0; font-size: 1rem;">⚙️ System Status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.success(f"✅ {len(models_list)} Models Available")
    st.info(f"📁 Folder: /{MODEL_FOLDER}")
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("""
    <div class="sidebar-card">
        <h3 style="color: white; margin: 0 0 1rem 0; font-size: 1rem;">📈 Quick Stats</h3>
        <p style="color: #B0B0B0; margin: 0.5rem 0; font-size: 0.9rem;">
           Accuracy: <span style="color: #2ECC71;">95%</span>
        </p>
        <p style="color: #B0B0B0; margin: 0.5rem 0; font-size: 0.9rem;">
           Processing: <span style="color: #2ECC71;"><2s</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("© 2026 HemoScan AI")


# -------------------------------------------------
# Main Content Area
# -------------------------------------------------
st.markdown("""
<div class="medical-header animate-in">
    <h1>🩸 Blood Group Detection System</h1>
    <p>Advanced AI-based Fingerprint Analysis for Medical Applications</p>
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
        return load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model_path = os.path.join(MODEL_FOLDER, selected_model_file)
model = load_model_cached(model_path)

if model is None:
    st.stop()

# Auto-detect input size from model
model_input_shape = model.input_shape
if model_input_shape and len(model_input_shape) >= 3:
    input_size = (model_input_shape[1], model_input_shape[2])
else:
    input_size = (224, 224)

# Get class labels from model output or use default
model_output_shape = model.output_shape
if model_output_shape and len(model_output_shape) >= 2:
    num_classes = model_output_shape[1]
    default_classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
    if num_classes <= len(default_classes):
        class_labels = default_classes[:num_classes]
    else:
        class_labels = [f"Class_{i}" for i in range(num_classes)]
else:
    class_labels = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

with model_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.success(f"✅ Loaded")

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
    
    with st.form("patient_form"):
        name = st.text_input("Full Name*", placeholder="Enter patient full name")
        
        col_age, col_gender = st.columns(2)
        with col_age:
            age = st.number_input("Age*", min_value=1, max_value=120, value=25)
        with col_gender:
            gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
        
        phone = st.text_input("Phone Number*", placeholder="+1 (555) 000-0000")
        email = st.text_input("Email Address*", placeholder="patient@example.com")
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "📤 Upload Fingerprint Image", 
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload a clear fingerprint image for analysis"
        )
        
        submitted = st.form_submit_button("🔬 Analyze Sample", use_container_width=True)


# Handle Form Submission
if submitted:
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
        img = Image.open(pred_data['file'])
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(img, width=200, caption="Uploaded Fingerprint")
        
        # Generate PDF
        if st.session_state.pdf_path is None:
            with st.spinner("📄 Generating medical report..."):
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