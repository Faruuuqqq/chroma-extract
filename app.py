import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

st.set_page_config(page_title="ChromaExtract", layout="wide")

# Custom CSS - Light mode, compact, modern
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* Apply modern font */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Background */
.stApp {
    background: radial-gradient(circle at 10% 20%, #f8fafc 0%, #e2e8f0 100%);
    color: #0f172a;
}

.stTitle { 
    background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

.stSubtitle {
    color: #64748b;
    font-size: 1.2rem;
    font-weight: 500;
    margin-top: -10px;
    margin-bottom: 2rem;
}

/* Unified Palette Card */
.palette-container {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.01);
    border: 1px solid rgba(0,0,0,0.05);
    margin-bottom: 2rem;
}
.ribbon-container {
    display: flex;
    height: 120px;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1.5rem;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.1);
}
.ribbon-segment {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: crosshair;
}
.ribbon-segment:hover {
    flex: 1.5;
}
.ribbon-text {
    font-size: 0.85rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    opacity: 0;
    transition: opacity 0.3s;
    background: rgba(255,255,255,0.4);
    padding: 4px 8px;
    border-radius: 6px;
    backdrop-filter: blur(4px);
}
.ribbon-segment:hover .ribbon-text {
    opacity: 1;
}
.details-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 1rem;
}
.detail-item {
    text-align: center;
    flex: 1 1 0px;
    min-width: 90px;
}

/* Clean up Streamlit uploader */
div[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.6);
    border-radius: 16px;
    padding: 2rem;
    border: 2px dashed #cbd5e1;
    transition: all 0.3s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #ec4899;
    background: rgba(255, 255, 255, 0.9);
}

/* Style Streamlit Tabs */
button[data-baseweb="tab"] {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    background-color: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #a855f7 !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #a855f7 !important;
}

/* Style the step-by-step text */
.step-box {
    background: rgba(255, 255, 255, 0.8);
    border-left: 4px solid #a855f7;
    padding: 1.5rem;
    border-radius: 0 12px 12px 0;
    margin: 1.5rem 0;
    border-top: 1px solid rgba(0,0,0,0.05);
    border-right: 1px solid rgba(0,0,0,0.05);
    border-bottom: 1px solid rgba(0,0,0,0.05);
}
.step-box p {
    color: #334155;
    font-size: 1.05rem;
    line-height: 1.6;
    margin-bottom: 0.8rem;
}
.step-box strong {
    color: #9333ea;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_dominant_colors(image, k=5):
    img = image.resize((150, 150))
    pixels = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(pixels)
    colors = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    return colors[np.argsort(-counts)].astype(int), kmeans, pixels

def visualize_kmeans_process(image, k=5):
    """Visualize K-Means clustering step by step"""
    img = image.resize((150, 150))
    pixels = np.array(img).reshape(-1, 3)
    
    # Initialize K-Means with n_init=1 for step-by-step visualization
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=1, max_iter=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#ffffff')
    
    # Step 1: Original Image
    ax = axes[0, 0]
    ax.imshow(img)
    ax.set_title("Step 1: Original Image (150×150px)", color='black', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Step 2: Pixel Distribution (RGB 3D scatter - simplified to 2D for viz)
    ax = axes[0, 1]
    ax.scatter(pixels[:, 0], pixels[:, 1], c=pixels/255, s=10, alpha=0.6)
    ax.set_xlabel('Red', color='black')
    ax.set_ylabel('Green', color='black')
    ax.set_title("Step 2: Pixel Distribution (R vs G)", color='black', fontsize=12, fontweight='bold')
    ax.tick_params(colors='black')
    ax.set_facecolor('#f8fafc')
    
    # Step 3: Initial Random Centers
    ax = axes[1, 0]
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_
    ax.scatter(pixels[:, 0], pixels[:, 1], c=pixels/255, s=5, alpha=0.3)
    ax.scatter(centers[:, 0], centers[:, 1], c=centers/255, s=300, marker='*', 
               edgecolors='black', linewidths=2, label='Cluster Centers')
    ax.set_xlabel('Red', color='black')
    ax.set_ylabel('Green', color='black')
    ax.set_title(f"Step 3: K-Means Converged (K={k})", color='black', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', labelcolor='black')
    ax.tick_params(colors='black')
    ax.set_facecolor('#f8fafc')
    
    # Step 4: Color Clusters
    ax = axes[1, 1]
    labels = kmeans.labels_
    ax.scatter(pixels[:, 0], pixels[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    ax.scatter(centers[:, 0], centers[:, 1], c=centers/255, s=300, marker='*',
               edgecolors='black', linewidths=2)
    ax.set_xlabel('Red', color='black')
    ax.set_ylabel('Green', color='black')
    ax.set_title("Step 4: Assigned Clusters", color='black', fontsize=12, fontweight='bold')
    ax.tick_params(colors='black')
    ax.set_facecolor('#f8fafc')
    
    plt.tight_layout()
    return fig, kmeans

def rgb_to_hex(rgb):
    return f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'

def rgb_to_hsl(rgb):
    r, g, b = rgb[0]/255, rgb[1]/255, rgb[2]/255
    mx, mn = max(r,g,b), min(r,g,b)
    l = (mx + mn) / 2
    if mx == mn:
        h = s = 0
    else:
        d = mx - mn
        s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
        if mx == r:
            h = (60 * ((g - b) / d) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / d) + 120) % 360
        else:
            h = (60 * ((r - g) / d) + 240) % 360
    return int(h), int(s*100), int(l*100)

# Main UI
st.markdown('<h1 class="stTitle">ChromaExtract</h1>', unsafe_allow_html=True)
st.markdown('<p class="stSubtitle">Premium Color Palette Generator powered by K-Means Clustering</p>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.markdown("### Configuration")
    k_colors = st.slider("Number of Colors (K)", min_value=2, max_value=10, value=5, step=1)
    st.markdown("---")
    st.markdown("### About")
    st.info("This tool uses **Machine Learning** (K-Means Clustering) to find the most dominant colors in any image.")

uploaded = st.file_uploader("Drop an image here to get started...", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    
    st.markdown("### Your Extracted Palette")
    
    with st.spinner("Analyzing pixels and clustering colors..."):
        colors, kmeans, pixels = get_dominant_colors(image, k=k_colors)
    
    # Unified Palette Card
    ribbon_html = ""
    details_html = ""
    
    for color in colors:
        hex_color = rgb_to_hex(color)
        h, s, l = rgb_to_hsl(color)
        rgb_str = f"{color[0]}, {color[1]}, {color[2]}"
        
        # Calculate contrast text color for the hex code on hover
        text_color = "#0f172a" if l > 65 else "#ffffff"
        
        ribbon_html += f'<div class="ribbon-segment" style="background-color: {hex_color};"><span class="ribbon-text" style="color: {text_color};">{hex_color}</span></div>'
        
        details_html += f'''
<div class="detail-item">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; color: #1e293b;">{hex_color}</div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #64748b; margin-top: 4px;">rgb({rgb_str})</div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">hsl({h}, {s}%, {l}%)</div>
</div>
'''

    full_card = f'''
<div class="palette-container">
    <div class="ribbon-container">
        {ribbon_html}
    </div>
    <div class="details-container">
        {details_html}
    </div>
</div>
'''
    st.markdown(full_card, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Stacked layout instead of columns
    st.markdown("### Original Image")
    st.image(image, width='stretch')
    st.caption(f"Resolution: {image.size[0]} × {image.size[1]}px | Total Pixels: {image.size[0] * image.size[1]:,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Export options
    with st.expander("Export Palette (HEX / CSS / JSON)"):
        export_tab1, export_tab2, export_tab3 = st.tabs(["HEX", "CSS", "JSON"])
        with export_tab1:
            hex_out = "\\n".join([rgb_to_hex(c) for c in colors])
            st.code(hex_out)
        with export_tab2:
            css_out = ":root {\\n" + "\\n".join([f"  --color-{i+1}: {rgb_to_hex(c)};" for i, c in enumerate(colors)]) + "\\n}"
            st.code(css_out, language="css")
        with export_tab3:
            json_out = json.dumps({f"color_{i+1}": {"hex": rgb_to_hex(c), "rgb": f"rgb({c[0]},{c[1]},{c[2]})"} for i, c in enumerate(colors)}, indent=2)
            st.code(json_out, language="json")
            
    st.markdown("<br>How it works: K-Means Clustering", unsafe_allow_html=True)
    st.markdown("""
    K-Means is an **unsupervised machine learning algorithm** that groups similar data points. 
    Here, data points are **pixels**, and they are grouped based on their **RGB color values**.
    """)
    
    tab_steps, tab_viz = st.tabs(["Step-by-Step Process", "Advanced Visualization"])
    
    with tab_steps:
        st.markdown("""
<div class="step-box" style="margin-bottom: 1.5rem;">
    <p><strong>How K-Means Learns:</strong> It iteratively finds the best central colors (centroids) that represent all pixels in your image.</p>
</div>
        """, unsafe_allow_html=True)
        
        # Using Streamlit columns instead of matplotlib for the 3 steps
        step1, step2, step3 = st.columns(3)
        
        with step1:
            st.markdown(f"""
<div style="background: rgba(255, 255, 255, 0.9); padding: 1.2rem; border-radius: 16px; border-top: 4px solid #3b82f6; height: 100%; box-shadow: 0 4px 10px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.05);">
    <h4 style="color: #2563eb; margin-bottom: 0.8rem; font-size: 1.1rem;">1. Data Prep</h4>
    <p style="color: #475569; font-size: 0.9rem; margin-bottom: 0.2rem;">• Resize image</p>
    <p style="color: #475569; font-size: 0.9rem; margin-bottom: 0.2rem;">• Extract RGB</p>
    <p style="color: #475569; font-size: 0.9rem; margin-bottom: 0.8rem;">• <b>{len(pixels):,}</b> pixels</p>
    <div style="background: #f8fafc; padding: 0.8rem; border-radius: 8px; border: 1px solid #e2e8f0;">
        <div style="color: #475569; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;">Sample: {pixels[0].tolist()}</div>
    </div>
</div>
            """, unsafe_allow_html=True)
            
        with step2:
            st.markdown(f"""
<div style="background: rgba(255, 255, 255, 0.9); padding: 1.2rem; border-radius: 16px; border-top: 4px solid #a855f7; height: 100%; box-shadow: 0 4px 10px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.05);">
    <h4 style="color: #9333ea; margin-bottom: 0.8rem; font-size: 1.1rem;">2. Clustering</h4>
    <p style="color: #475569; font-size: 0.9rem; margin-bottom: 0.2rem;">• Init {k_colors} centers</p>
    <p style="color: #475569; font-size: 0.9rem; margin-bottom: 0.2rem;">• Assign pixels</p>
    <p style="color: #475569; font-size: 0.9rem; margin-bottom: 0.8rem;">• Update centers</p>
    <div style="background: #f8fafc; padding: 0.8rem; border-radius: 8px; border: 1px solid #e2e8f0;">
        <div style="color: #475569; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;">Iter: {kmeans.n_iter_} | In: {kmeans.inertia_:.0f}</div>
    </div>
</div>
            """, unsafe_allow_html=True)
            
        with step3:
            centers_list = "<br>".join([f"<span style='color: rgb({c[0]},{c[1]},{c[2]}); font-size: 1.2rem; vertical-align: middle;'>■</span> rgb({c[0]}, {c[1]}, {c[2]})" for c in kmeans.cluster_centers_.astype(int)[:3]])
            if len(kmeans.cluster_centers_) > 3: centers_list += "<br><span style='color: #94a3b8;'>...and more</span>"
            st.markdown(f"""
<div style="background: rgba(255, 255, 255, 0.9); padding: 1.2rem; border-radius: 16px; border-top: 4px solid #ec4899; height: 100%; box-shadow: 0 4px 10px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.05);">
    <h4 style="color: #db2777; margin-bottom: 0.8rem; font-size: 1.1rem;">3. Results</h4>
    <p style="color: #475569; font-size: 0.9rem; margin-bottom: 0.2rem;">• Get centroids</p>
    <p style="color: #475569; font-size: 0.9rem; margin-bottom: 0.2rem;">• Sort colors</p>
    <p style="color: #475569; font-size: 0.9rem; margin-bottom: 0.8rem;">• Done!</p>
    <div style="background: #f8fafc; padding: 0.8rem; border-radius: 8px; border: 1px solid #e2e8f0;">
        <div style="color: #475569; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; line-height: 1.4;">{centers_list}</div>
    </div>
</div>
            """, unsafe_allow_html=True)
        
    with tab_viz:
        st.markdown("""
<div class="step-box" style="border-left-color: #ec4899;">
    <p style="margin-bottom:0;">This shows how pixels are distributed in a 2D color space (Red vs Green) and how K-Means clusters them.</p>
</div>
        """, unsafe_allow_html=True)
        with st.spinner("Generating visualization..."):
            fig_viz, _ = visualize_kmeans_process(image, k=k_colors)
            st.pyplot(fig_viz)

else:
    # Empty state styling
    st.markdown("""
<div style="text-align: center; padding: 5rem 2rem; background: rgba(255, 255, 255, 0.8); border-radius: 20px; border: 2px dashed #cbd5e1; margin-top: 2rem; backdrop-filter: blur(10px);">
    <h3 style="color: #334155; font-size: 1.8rem; margin-bottom: 1rem;">No Image Uploaded</h3>
    <p style="color: #64748b; font-size: 1.1rem;">Upload an image above to extract a beautiful color palette and see how AI does it under the hood!</p>
</div>
    """, unsafe_allow_html=True)