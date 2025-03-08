#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import os
import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import CharacterTextSplitter
import re
import PyPDF2
from io import BytesIO

# Fetch API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY is not set. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=api_key)  # Use the API key from the environment

# Initialize the Gemini model
def get_model():
    return genai.GenerativeModel("gemini-2.0-flash")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Process long text by splitting and summarizing
def process_long_text(text):
    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # For multiple chunks, summarize first
    if len(chunks) > 1:
        model = get_model()
        summaries = []
        for chunk in chunks:
            response = model.generate_content(f"Summarize the following text concisely:\n\n{chunk}")
            summaries.append(response.text)
        
        return " ".join(summaries)
    else:
        return text

# Function to generate SWOT analysis
def generate_swot_analysis(text):
    # Process long text first
    processed_text = process_long_text(text)
    
    model = get_model()
    prompt = f"""
    Based on the following text, create a comprehensive SWOT analysis:
    
    {processed_text}
    
    Format your response as follows:

    STRENGTHS:
    • [Strength 1]: [Brief explanation]
    • [Strength 2]: [Brief explanation]
    • [Strength 3]: [Brief explanation]
    • [Strength 4]: [Brief explanation]
    • [Strength 5]: [Brief explanation]

    WEAKNESSES:
    • [Weakness 1]: [Brief explanation]
    • [Weakness 2]: [Brief explanation]
    • [Weakness 3]: [Brief explanation]
    • [Weakness 4]: [Brief explanation]
    • [Weakness 5]: [Brief explanation]

    OPPORTUNITIES:
    • [Opportunity 1]: [Brief explanation]
    • [Opportunity 2]: [Brief explanation]
    • [Opportunity 3]: [Brief explanation]
    • [Opportunity 4]: [Brief explanation]
    • [Opportunity 5]: [Brief explanation]

    THREATS:
    • [Threat 1]: [Brief explanation]
    • [Threat 2]: [Brief explanation]
    • [Threat 3]: [Brief explanation]
    • [Threat 4]: [Brief explanation]
    • [Threat 5]: [Brief explanation]

    Use bullet points exactly as shown above with the • symbol. Be specific and concise with each point.
    """
    
    response = model.generate_content(prompt)
    return response.text

# Function to generate insights from SWOT analysis
def generate_swot_insights(swot_text):
    model = get_model()
    
    prompt = f"""
    Based on the following SWOT analysis, provide strategic insights and recommendations:
    
    {swot_text}
    
    Please provide:
    1. Three key strategic insights based on this SWOT analysis
    2. Three actionable recommendations for leveraging strengths and opportunities
    3. Two mitigation strategies for addressing weaknesses and threats
    4. A recommended priority focus area
    
    Format your response in bullet points with clear sections.
    """
    
    response = model.generate_content(prompt)
    return response.text

# Function to parse SWOT analysis into sections
def parse_swot_analysis(swot_text):
    sections = {
        'strengths': [],
        'weaknesses': [],
        'opportunities': [],
        'threats': []
    }
    
    # Use regex to extract sections
    patterns = {
        'strengths': r'STRENGTHS:(.+?)(?=WEAKNESSES:|OPPORTUNITIES:|THREATS:|$)',
        'weaknesses': r'WEAKNESSES:(.+?)(?=STRENGTHS:|OPPORTUNITIES:|THREATS:|$)',
        'opportunities': r'OPPORTUNITIES:(.+?)(?=STRENGTHS:|WEAKNESSES:|THREATS:|$)',
        'threats': r'THREATS:(.+?)(?=STRENGTHS:|WEAKNESSES:|OPPORTUNITIES:|$)'
    }
    
    for section, pattern in patterns.items():
        matches = re.search(pattern, swot_text, re.DOTALL | re.IGNORECASE)
        if matches:
            # Extract bullet points
            content = matches.group(1).strip()
            bullet_items = re.findall(r'•\s*(.*?)(?=•|\Z)', content + '•', re.DOTALL)
            
            # Clean up bullet points
            sections[section] = [item.strip() for item in bullet_items if item.strip()]
            
            # Fallback: If no bullet points found, try line-by-line
            if not sections[section]:
                lines = content.split('\n')
                sections[section] = [line.strip().lstrip('-•').strip() for line in lines 
                                    if line.strip() and not line.strip().startswith('STRENGTHS:') 
                                    and not line.strip().startswith('WEAKNESSES:') 
                                    and not line.strip().startswith('OPPORTUNITIES:') 
                                    and not line.strip().startswith('THREATS:')]
    
    return sections

# Function to format SWOT analysis in a box pattern
def format_swot_box(sections):
    html = """
    <style>
        .swot-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto auto;
            gap: 15px;
            margin-top: 20px;
        }
        .swot-box {
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            min-height: 250px;
        }
        .strengths {
            background-color: #1e4620;
            border-left: 5px solid #28a745;
            color: #fff;
        }
        .weaknesses {
            background-color: #5c1624;
            border-left: 5px solid #dc3545;
            color: #fff;
        }
        .opportunities {
            background-color: #0c4271;
            border-left: 5px solid #007bff;
            color: #fff;
        }
        .threats {
            background-color: #693c11;
            border-left: 5px solid #ffc107;
            color: #fff;
        }
        .swot-title {
            font-weight: bold;
            font-size: 1.4em;
            margin-bottom: 15px;
            color: #ffffff;
        }
        .swot-content {
            font-size: 1em;
            color: #f0f0f0;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
    </style>
    
    <div class="swot-container">
    """
    
    # Add each section to the HTML
    sections_data = [
        ('strengths', 'Strengths', 'strengths'),
        ('weaknesses', 'Weaknesses', 'weaknesses'),
        ('opportunities', 'Opportunities', 'opportunities'),
        ('threats', 'Threats', 'threats')
    ]
    
    for key, title, css_class in sections_data:
        html += f'<div class="swot-box {css_class}"><div class="swot-title">{title}</div><div class="swot-content">'
        if key in sections and sections[key]:
            html += '<ul>'
            for item in sections[key]:
                html += f'<li>{item}</li>'
            html += '</ul>'
        else:
            html += 'No data available.'
        html += '</div></div>'
    
    html += '</div>'
    return html

# Function to format insights without BeautifulSoup
def format_insights(insights_text):
    # Simple HTML escape function for safety
    def html_escape(text):
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
    
    # Safely convert newlines into HTML tags
    escaped_content = html_escape(insights_text)
    content = escaped_content.replace('\n\n', '</div><div class="insight-section">')
    content = content.replace('\n', '<br>')
    content = f'<div class="insight-section">{content}</div>'
    
    # HTML structure for the insights section with darker styling
    html = """
    <style>
        .insights-container {
            margin-top: 30px;
            padding: 20px;
            background-color: #333;
            border-radius: 8px;
            border-left: 5px solid #6610f2;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .insights-title {
            font-weight: bold;
            font-size: 1.4em;
            margin-bottom: 15px;
            color: #fff;
        }
        .insights-content {
            font-size: 1em;
            color: #fff;
        }
        .insight-section {
            margin-bottom: 15px;
        }
        .section-title {
            font-weight: bold;
            margin-bottom: 8px;
            color: #ccc;
        }
    </style>
    
    <div class="insights-container">
        <div class="insights-title">Strategic Insights &amp; Recommendations</div>
        <div class="insights-content">
            {content}
        </div>
    </div>
    """
    
    # Replace the placeholder with the sanitized content
    return html.replace("{content}", content)

# Streamlit UI Setup
st.set_page_config(page_title="SWOT Analysis Generator", layout="wide")

# Add a custom header with styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(to right, #4880EC, #019CAD);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.2em;
        opacity: 0.8;
    }
    </style>
    <div class="main-header">
        <h1>SWOT Analysis Generator</h1>
        <p class="sub-header">Powered by Google Gemini AI</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["PDF Upload", "Text Input", "About"])

# PDF Upload Tab
with tab1:
    st.write("Upload a PDF document to generate a SWOT analysis.")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")
    
    if uploaded_file is not None:
        with st.spinner('Processing PDF...'):
            pdf_text = extract_text_from_pdf(BytesIO(uploaded_file.getvalue()))
            
            # Show a preview of the extracted text
            with st.expander("Preview of Extracted Text"):
                st.text(pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text)
            
        if st.button("Generate SWOT Analysis", key="generate_pdf"):
            with st.spinner('Generating SWOT Analysis...'):
                swot_analysis = generate_swot_analysis(pdf_text)
                
                # Parse SWOT analysis
                swot_sections = parse_swot_analysis(swot_analysis)
                
                # Display the raw SWOT analysis
                with st.expander("Raw SWOT Analysis"):
                    st.markdown(swot_analysis)
                
                # Display the formatted SWOT box
                st.subheader("SWOT Analysis")
                formatted_swot = format_swot_box(swot_sections)
                st.markdown(formatted_swot, unsafe_allow_html=True)
                
                # Generate and display insights
                with st.spinner('Generating Strategic Insights...'):
                    insights = generate_swot_insights(swot_analysis)
                    
                    st.subheader("Strategic Insights")
                    with st.expander("Raw Insights"):
                        st.markdown(insights)
                        
                    formatted_insights = format_insights(insights)
                    st.markdown(formatted_insights, unsafe_allow_html=True)
                
                # Option to download the complete analysis
                combined_analysis = f"SWOT ANALYSIS\n\n{swot_analysis}\n\nSTRATEGIC INSIGHTS\n\n{insights}"
                
                st.download_button(
                    label="Download Complete Analysis",
                    data=combined_analysis,
                    file_name="swot_analysis.txt",
                    mime="text/plain"
                )

# Text Input Tab
with tab2:
    st.write("Enter text directly to generate a SWOT analysis.")
    text_input = st.text_area("Enter text about a business, project, or situation:", height=200)
    
    if text_input:
        if st.button("Generate SWOT Analysis", key="generate_text"):
            with st.spinner('Generating SWOT Analysis...'):
                swot_analysis = generate_swot_analysis(text_input)
                
                # Parse SWOT analysis
                swot_sections = parse_swot_analysis(swot_analysis)
                
                # Display the raw SWOT analysis
                with st.expander("Raw SWOT Analysis"):
                    st.markdown(swot_analysis)
                
                # Display the formatted SWOT box
                st.subheader("SWOT Analysis")
                formatted_swot = format_swot_box(swot_sections)
                st.markdown(formatted_swot, unsafe_allow_html=True)
                
                # Generate and display insights
                with st.spinner('Generating Strategic Insights...'):
                    insights = generate_swot_insights(swot_analysis)
                    
                    st.subheader("Strategic Insights")
                    with st.expander("Raw Insights"):
                        st.markdown(insights)
                        
                    formatted_insights = format_insights(insights)
                    st.markdown(formatted_insights, unsafe_allow_html=True)
                
                # Option to download the complete analysis
                combined_analysis = f"SWOT ANALYSIS\n\n{swot_analysis}\n\nSTRATEGIC INSIGHTS\n\n{insights}"
                
                st.download_button(
                    label="Download Complete Analysis",
                    data=combined_analysis,
                    file_name="swot_analysis.txt",
                    mime="text/plain"
                )

# About Tab
with tab3:
    st.markdown("""
    ## About SWOT Analysis Generator
    
    This application uses Google's Gemini AI to generate comprehensive SWOT (Strengths, Weaknesses, Opportunities, Threats) analyses and strategic insights from documents or text input.
    
    ### What is SWOT Analysis?
    
    SWOT Analysis is a strategic planning technique used to help identify Strengths, Weaknesses, Opportunities, and Threats related to business competition or project planning. It is designed to specify the objectives of the business venture or project and identify the internal and external factors that are favorable and unfavorable to achieving those objectives.
    
    ### How to Use This Tool
    
    1. **PDF Upload Tab**: Upload a PDF document containing information about a business, project, or situation
    2. **Text Input Tab**: Enter text directly about the subject you want to analyze
    3. **Generate Analysis**: Click the "Generate SWOT Analysis" button to process your input
    4. **Review Results**: Examine the formatted SWOT analysis and strategic insights
    5. **Download**: Save the complete analysis as a text file for future reference
    
    ### Powered by Gemini AI
    
    This application leverages Google's Gemini AI model to analyze your input and generate meaningful insights. The AI processes the content, identifies key elements, and organizes them into a structured SWOT framework.
    """)

# Add a footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: #6c757d;
        text-align: center;
        padding: 10px;
        font-size: 0.8em;
    }
    </style>
    <div class="footer">
        SWOT Analysis Generator © 2025 | Powered by Google Gemini AI
    </div>
""", unsafe_allow_html=True)

