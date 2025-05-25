import streamlit as st
import pandas as pd
import numpy as np
import base64
import joblib
import plotly.express as px
import os
import traceback
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# -------------------- Page Config -------------------- #
st.set_page_config(
    page_title="IoT IDS ML Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# -------------------- Cyber Theme -------------------- #
st.markdown("""
    <!-- Google Font: Poppins -->
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">

    <style>
    .appview-container {
        background: 
            linear-gradient(rgba(15, 17, 23, 0.75), rgba(15, 17, 23, 0.75)),
            url('https://imageio.forbes.com/specials-images/imageserve/6436b538888732e60f6ba1ae//960x0.jpg?height=474&width=711&fit=bounds%27');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
        font-size: 16px !important;
        color: white !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, div {
        color: white !important;
    }

    .main {
        background-color: transparent;
    }

    .stButton>button, a > button {
        background-color: #003366 !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        border: none !important;
        padding: 10px 20px !important;
        cursor: pointer !important;
    }

    a > button:hover {
        background-color: #001f4d !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Header -------------------- #
st.markdown("<h1 style='text-align: center; color: white;'> ML-based Intrusion Detection System for IoT Networks</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>Protect your network with the power of Machine Learning 🛡️</h4>", unsafe_allow_html=True)

# -------------------- About the Project -------------------- #
st.markdown("---")
st.subheader("📘 About the Project")
st.markdown("""
**This project presents a comprehensive study on using Machine Learning (ML) algorithms to build an efficient Intrusion Detection System (IDS) tailored for Internet of Things (IoT) networks. By leveraging the recent and realistic CICIoT2023 dataset, the study evaluates multiple supervised ML models under various preprocessing and balancing strategies. The goal is to identify lightweight and accurate models that can detect different types of cyberattacks in resource-constrained IoT environments. Among the evaluated models, XGBoost achieved the highest performance, making it the final selected model for reliable and scalable intrusion detection in IoT settings**.
""")
st.markdown("---")

# -------------------- Load Pretrained Model & Label Encoder -------------------- #
model_path = "xgb_model.pkl"
encoder_path = "label_encoder.pkl"

#st.write("Model file exists:", os.path.exists(model_path))
#st.write("Encoder file exists:", os.path.exists(encoder_path))

#st.write("Model file readable:", os.access(model_path, os.R_OK))
#st.write("Encoder file readable:", os.access(encoder_path, os.R_OK))

try:
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
except Exception as e:
    st.error(f"❌ Failed to load model or encoder: {e}")
    st.error(traceback.format_exc())
    st.stop()

# -------------------- Attack Label Mapping -------------------- #
attack_labels = {
    0: ('Backdoor_Malware', 'Malware that allows remote control of infected systems'),
    1: ('BenignTraffic', 'Normal, harmless network traffic'),
    2: ('BrowserHijacking', 'Alters browser behavior or redirects to malicious sites'),
    3: ('CommandInjection', 'Executes system commands via vulnerable inputs'),
    4: ('DDoS-ACK_Fragmentation', 'Distributed attack using fragmented ACK packets'),
    5: ('DDoS-HTTP_Flood', 'Overwhelms server with HTTP requests from multiple sources'),
    6: ('DDoS-ICMP_Flood', 'Floods network with ICMP echo requests (ping)'),
    7: ('DDoS-ICMP_Fragmentation', 'Uses fragmented ICMP packets to disrupt systems'),
    8: ('DDoS-PSHACK_Flood', 'Floods with TCP packets having PSH and ACK flags set'),
    9: ('DDoS-RSTFINFlood', 'TCP flood using RST and FIN flags to confuse target'),
    10: ('DDoS-SYN_Flood', 'Exploits TCP handshake with many SYN requests'),
    11: ('DDoS-SlowLoris', 'Keeps many connections open to exhaust server resources'),
    12: ('DDoS-SynonymousIP_Flood', 'Flooding using packets from spoofed IPs'),
    13: ('DDoS-TCP_Flood', 'Sends overwhelming TCP traffic to the target'),
    14: ('DDoS-UDP_Flood', 'Floods server with UDP packets to consume resources'),
    15: ('DDoS-UDP_Fragmentation', 'Uses fragmented UDP packets to crash systems'),
    16: ('DNS_Spoofing', 'Fakes DNS responses to redirect traffic'),
    17: ('DictionaryBruteForce', 'Attempts logins using a dictionary of passwords'),
    18: ('DoS-HTTP_Flood', 'Single-source HTTP request flooding'),
    19: ('DoS-SYN_Flood', 'Exploits SYN packets to deplete server resources'),
    20: ('DoS-TCP_Flood', 'TCP flooding attack from one source'),
    21: ('DoS-UDP_Flood', 'Sends large number of UDP packets to crash server'),
    22: ('MITM-ArpSpoofing', 'Intercepts data by poisoning ARP cache'),
    23: ('Mirai-greeth_flood', 'Mirai botnet GRE Ethernet flooding'),
    24: ('Mirai-greip_flood', 'Mirai botnet GRE IP flooding'),
    25: ('Mirai-udpplain', 'Basic Mirai botnet UDP flooding'),
    26: ('Recon-HostDiscovery', 'Detects live hosts on the network'),
    27: ('Recon-OSScan', 'Identifies OS details of target machines'),
    28: ('Recon-PingSweep', 'Scans for active hosts using ICMP'),
    29: ('Recon-PortScan', 'Scans for open ports to find vulnerabilities'),
    30: ('SqlInjection', 'Injects SQL code to manipulate database'),
    31: ('Uploading_Attack', 'Uploads malicious files to the target'),
    32: ('VulnerabilityScan', 'Scans systems for known vulnerabilities'),
    33: ('XSS', 'Injects malicious scripts into web pages'),
}

# -------------------- Recommendations Mapping -------------------- #
attack_recommendations = {
    "Backdoor_Malware": "This attack allows remote access. Use endpoint protection and regularly audit your systems.",
    "BrowserHijacking": "Ensure browser extensions are vetted. Use secure DNS and ad blockers.",
    "CommandInjection": "Sanitize all user inputs and implement input validation.",
    "DDoS-ACK_Fragmentation": "Use DDoS mitigation services and configure firewalls to block fragmented packets.",
    "DDoS-HTTP_Flood": "Rate-limit HTTP requests and use load balancers with DDoS protection.",
    "DDoS-ICMP_Flood": "Limit ICMP requests and use intrusion prevention systems.",
    "DDoS-ICMP_Fragmentation": "Use advanced packet inspection and configure firewalls to drop malformed packets.",
    "DDoS-PSHACK_Flood": "Use traffic pattern analysis tools and TCP flood protection mechanisms.",
    "DDoS-RSTFINFlood": "Implement connection rate limiting and deep packet inspection.",
    "DDoS-SYN_Flood": "Deploy SYN cookies and use hardware firewalls.",
    "DDoS-SlowLoris": "Limit concurrent connections and enforce timeouts.",
    "DDoS-SynonymousIP_Flood": "Detect spoofed IPs using IP traceback and anomaly detection systems.",
    "DDoS-TCP_Flood": "Apply TCP flood detection tools and set rate limits.",
    "DDoS-UDP_Flood": "Filter UDP packets and use anomaly-based detection.",
    "DDoS-UDP_Fragmentation": "Block fragmented UDP packets and monitor for unusual traffic sizes.",
    "DNS_Spoofing": "Use DNSSEC and configure your DNS servers securely.",
    "DictionaryBruteForce": "Use account lockouts and multi-factor authentication.",
    "DoS-HTTP_Flood": "Use reverse proxies and request throttling.",
    "DoS-SYN_Flood": "Enable SYN cookies and limit connection attempts.",
    "DoS-TCP_Flood": "Configure TCP flood prevention settings in firewalls.",
    "DoS-UDP_Flood": "Limit UDP traffic rate and use detection tools.",
    "MITM-ArpSpoofing": "Use encrypted protocols (HTTPS, SSH) and enable dynamic ARP inspection.",
    "Mirai-greeth_flood": "Use GRE filtering rules and botnet detection systems.",
    "Mirai-greip_flood": "Block unwanted GRE traffic and monitor unusual patterns.",
    "Mirai-udpplain": "Detect botnet behavior and filter by signature-based systems.",
    "Recon-HostDiscovery": "Block ICMP echo requests and use intrusion detection systems.",
    "Recon-OSScan": "Disable OS fingerprinting responses and use honeypots.",
    "Recon-PingSweep": "Monitor for large-scale ICMP activity and block unknown sources.",
    "Recon-PortScan": "Use port scan detection tools and close unused ports.",
    "SqlInjection": "Use parameterized queries and validate user inputs.",
    "Uploading_Attack": "Restrict file types and scan uploads with antivirus software.",
    "VulnerabilityScan": "Patch systems regularly and run internal scans.",
    "XSS": "Escape HTML output and implement Content Security Policy (CSP)."
}

# -------------------- Upload Dataset without Label (For Prediction) -------------------- #
st.subheader("📂 Upload Dataset (features only, no label)")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="no_label")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # Transform object columns using the preloaded encoder (if available)
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = label_encoder.transform(data[col].astype(str))

        st.write("### Dataset Preview:")
        st.dataframe(data.head())

        if st.button("🚀 Predict Attacks", key="predict_no_label"):
            preds = model.predict(data)
            preds_named = [(attack_labels.get(label, ("Unknown", ""))[0],
                            attack_labels.get(label, ("Unknown", ""))[1])
                           for label in preds]
            st.success("✅ Prediction Completed!")
            st.write("### Predicted Attack Classes:")

            results_with_recommendations = []
            for label in preds:
                attack_name, description = attack_labels.get(label, ("Unknown", ""))
                recommendation = attack_recommendations.get(attack_name, "No recommendation available.")
                results_with_recommendations.append({
                    "Predicted Attack": attack_name,
                    "Description": description,
                    "Recommendation": recommendation
                })

            st.dataframe(pd.DataFrame(results_with_recommendations))

            # -------- Pie Chart: Attack Type Distribution -------- #
            attack_names = [res["Predicted Attack"] for res in results_with_recommendations]
            attack_counts = pd.Series(attack_names).value_counts().reset_index()
            attack_counts.columns = ['Attack Type', 'Count']

            import plotly.express as px

            fig_pie = px.pie(
                attack_counts,
                names='Attack Type',
                values='Count',
                title="🔍 Distribution of Detected Attack Types",
                color_discrete_sequence=px.colors.qualitative.Pastel,  # ألوان ناعمة
                hole=0.3
            )

            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',  # إزالة خلفية الرسم
                plot_bgcolor='rgba(0,0,0,0)',   # إزالة خلفية المخطط
                title_font_size=20,
                legend_title="",
                legend_font_size=12
            )

            st.markdown("### 📊 Attack Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
            st.subheader("⚠️ Risk Score & Security Recommendations")

            total = len(preds)
            attacks_count = sum([1 for p in preds if p != 1])
            risk_score = round((attacks_count / total) * 100, 2)

            st.markdown(f"**Risk Score:** {risk_score}%")

            if risk_score == 0:
                st.success("Your network is currently safe. Keep monitoring regularly.")
            elif risk_score <= 20:
                st.info("Low risk detected. Review activities and update firewall rules.")
            elif risk_score <= 50:
                st.warning("Moderate risk. Patch vulnerabilities and monitor actively.")
            else:
                st.error("High risk! Take immediate action to secure your network.")

    except Exception as e:
        st.error(f"❌ Error reading or processing file: {e}")


# -------------------- Upload Dataset with Label (For Accuracy Calculation) -------------------- #
st.markdown("---")
st.subheader("📂 Upload Dataset with Labels (for accuracy check)")
uploaded_file_with_label = st.file_uploader("Upload CSV File with label column (named exactly 'label') - Optional for testing and accuracy evaluation", type=["csv"], key="with_label")

if uploaded_file_with_label is not None:
    try:
        data_label = pd.read_csv(uploaded_file_with_label)

        if 'label' not in data_label.columns:
            st.error("❌ The uploaded file must contain a 'label' column.")
        else:
            X = data_label.drop(columns=['label'])
            y_true = data_label['label']

            # Encode object columns if any in X
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = label_encoder.transform(X[col].astype(str))

            # If labels are string, encode them as well for comparison
            if y_true.dtype == object:
                y_true_encoded = label_encoder.transform(y_true.astype(str))
            else:
                y_true_encoded = y_true.values

            st.write("### Dataset Preview:")
            st.dataframe(data_label.head())

            if st.button("🚀 Predict & Calculate Accuracy", key="predict_with_label"):
                preds = model.predict(X)

                # Accuracy calculation
                accuracy = np.mean(preds == y_true_encoded) * 100
                st.success(f"✅ Prediction Completed! Accuracy: {accuracy:.2f}%")

                # Show sample prediction results with labels
                preds_named = [attack_labels.get(label, ("Unknown", ""))[0] for label in preds]
                y_true_named = [attack_labels.get(label, ("Unknown", ""))[0] for label in y_true_encoded]

                results_df = pd.DataFrame({
                    'True Label': y_true_named,
                    'Predicted Label': preds_named
                })

                st.write("### Sample Predictions vs True Labels:")
                st.dataframe(results_df.head(20))

            if st.button("🔄 Reset Page", key="reset_with_label"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                # Reset file_uploader by rerunning the app
                st.rerun()


    except Exception as e:
        st.error(f"Error reading or processing file: {e}")

# -------------------- Download PDF -------------------- #
import streamlit as st

# -------------------- Download Section -------------------- #
st.markdown("---")
st.subheader("Download Our Study")
drive_link = "https://drive.google.com/uc?export=download&id=14Z4E_bcicpNCjTq6olOvUwT4ZT_zYYJt"
st.markdown(f"""
    <div>
        <a href="{drive_link}" target="_blank">
            <button style="
                background-color: #0b84ff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
            ">
                📥 Download Documentation
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)

# -------------------- Our Team -------------------- #
st.markdown("---")
st.subheader("Our Team")
def circular_image(image_url, name, linkedin_url, email):
    st.markdown(f"""
    <div style="text-align: center;">
        <img src="{image_url}" 
             style="border-radius: 50%; width: 150px; height: 150px; object-fit: cover;" />
        <div style="margin-top: 10px; font-weight: bold; color: white;">{name}</div>
        <div style="margin-top: 8px;">
            <span style="display: inline-block; margin-right: 10px;">
                <a href="{linkedin_url}" target="_blank">
                    <img src="https://i.imgur.com/YoHLlN8.png" 
                         width="24" height="24" style="vertical-align: middle;" />
                </a>
            </span>
            <span style="display: inline-block;">
                <a href="mailto:{email}">
                    <img src="https://i.imgur.com/pPih0Qn.png" 
                         width="24" height="24" style="vertical-align: middle;" />
                </a>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Team members with image, LinkedIn, and email
teammates = [
    ("https://i.imgur.com/sD4pIHh.jpeg", "Rama Alamaireh", "https://www.linkedin.com/in/rama-alamaireh/", "ramaalamairh0909@gmail.com"),
    ("https://i.imgur.com/X7CCrrh.jpeg", "Sewar Ismail", "https://www.linkedin.com/in/sewar-ismael-9b6a1528b/", "sewarismael2003@gmail.com"),
    ("https://i.imgur.com/4Af1aFP.jpeg", "Shahd Aljamal", "https://www.linkedin.com/in/shahd-abdallah/", "abdallahshahd47@gmail.com")
]

# --- Layout: Triangle ---
col_top = st.columns(3)
with col_top[1]:
    circular_image(*teammates[0])

col_bottom = st.columns(2)
with col_bottom[0]:
    circular_image(*teammates[1])
with col_bottom[1]:
    circular_image(*teammates[2])

# -------------------- Footer -------------------- #
st.markdown("""
---
<center style="color: white;">
    Spaghetti Team | Cybersecurity x AI 🚀  
    <br>
     <a href="mailto:spaghettiteamcy@gmail.com" style="color: #00c0ff;">spaghettiteamcy@gmail.com</a>
</center>

""", unsafe_allow_html=True)
