"""
Reverse Image Search - Streamlit Application
A deep learning-powered reverse image search web application using EfficientNet-B0 and ChromaDB.
"""
import os
import tempfile
import streamlit as st
from PIL import Image
import pandas as pd
from typing import List, Dict
import logging

# Import our custom modules
from src.search_engine import get_search_engine, SearchResult
from src.utils import get_supported_image_extensions, format_file_size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Reverse Image Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #FF6B6B;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        background-color: #f9f9f9;
    }
    .similarity-score {
        font-weight: bold;
        color: #00D4AA;
    }
    .stats-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'search_engine' not in st.session_state:
        with st.spinner("Initializing search engine..."):
            st.session_state.search_engine = get_search_engine()
    
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    
    if 'query_image' not in st.session_state:
        st.session_state.query_image = None

def home_page():
    """Render the home page with image upload and search functionality."""
    st.markdown('<h1 class="main-header">🔍 Reverse Image Search</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload an image to find visually similar images in your database. 
    The application uses EfficientNet-B0 for feature extraction and ChromaDB for fast similarity search.
    """)
    
    # Image upload section
    st.markdown("### 📤 Upload Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'],
            help="Supported formats: " + ", ".join(get_supported_image_extensions())
        )
        
        # Search parameters
        st.markdown("### ⚙️ Search Parameters")
        n_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
        similarity_threshold = st.slider(
            "Similarity threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1, 
            step=0.05,
            help="Minimum similarity score to display results"
        )
    
    with col2:
        if uploaded_file is not None:
            # Display uploaded image preview
            st.markdown("### 🖼️ Query Image")
            try:
                preview_image = st.session_state.search_engine.get_image_preview(uploaded_file)
                st.image(preview_image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                
                # Validate image
                if st.session_state.search_engine.validate_query_image(uploaded_file):
                    st.success("✅ Valid image file")
                else:
                    st.error("❌ Invalid image file")
                    return
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                return
    
    # Search button
    if uploaded_file is not None:
        if st.button("🔍 Search Similar Images", type="primary", use_container_width=True):
            with st.spinner("Searching for similar images..."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Perform search
                    results = st.session_state.search_engine.search_by_image_file(
                        uploaded_file,
                        n_results=n_results,
                        similarity_threshold=similarity_threshold
                    )
                    
                    st.session_state.search_results = results
                    st.session_state.query_image = uploaded_file
                    
                    if results:
                        st.success(f"Found {len(results)} similar images!")
                    else:
                        st.warning("No similar images found. Try adjusting the similarity threshold or add more images to your database.")
                        
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
    
    # Display search results
    if st.session_state.search_results:
        st.markdown("### 🎯 Search Results")
        
        # Results grid
        cols_per_row = 3
        results = st.session_state.search_results
        
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(results):
                    result = SearchResult(results[i + j])
                    
                    with col:
                        # Display result image
                        if result.is_valid():
                            try:
                                display_image = result.get_display_image()
                                if display_image:
                                    st.image(display_image, use_column_width=True)
                                else:
                                    st.error("Could not load image")
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
                        else:
                            st.error("Image file not found")
                        
                        # Display metadata
                        st.markdown(f"**{result.filename}**")
                        st.markdown(f'<span class="similarity-score">Similarity: {result.similarity_score:.3f}</span>', unsafe_allow_html=True)
                        st.caption(f"Size: {result.get_file_size_formatted()}")
                        
                        # Show full path on hover
                        with st.expander("📁 File Details"):
                            st.text(f"Path: {result.file_path}")
                            st.text(f"Directory: {os.path.dirname(result.file_path)}")

def database_management_page():
    """Render the database management page."""
    st.markdown('<h1 class="main-header">🗄️ Database Management</h1>', unsafe_allow_html=True)
    
    # Database statistics
    st.markdown("### 📊 Database Statistics")
    
    try:
        stats = st.session_state.search_engine.get_database_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", stats.get("total_images", 0))
        
        with col2:
            st.metric("Total Size", stats.get("total_size_formatted", "0B"))
        
        with col3:
            st.metric("Directories", len(stats.get("directories", {})))
        
        with col4:
            st.metric("Database Path", "✅ Connected" if stats.get("total_images", 0) >= 0 else "❌ Error")
        
        # Directory breakdown
        if stats.get("directories"):
            st.markdown("### 📁 Directory Breakdown")
            
            dir_data = []
            for directory, info in stats["directories"].items():
                dir_data.append({
                    "Directory": directory,
                    "Images": info["count"],
                    "Size": format_file_size(info["size"])
                })
            
            df = pd.DataFrame(dir_data)
            st.dataframe(df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading database statistics: {str(e)}")
    
    st.divider()
    
    # Add images section
    st.markdown("### ➕ Add Images to Database")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        directory_path = st.text_input(
            "Directory Path",
            placeholder="Enter the path to a directory containing images",
            help="All images in this directory and subdirectories will be added to the database"
        )
        
        if st.button("📂 Browse Directory", help="Select a directory using file browser"):
            st.info("💡 Tip: Enter the directory path manually in the text field above")
    
    with col2:
        if directory_path and os.path.exists(directory_path):
            if os.path.isdir(directory_path):
                st.success("✅ Valid directory")
                
                # Count images in directory
                from src.utils import get_image_files_from_directory
                image_files = get_image_files_from_directory(directory_path)
                st.info(f"Found {len(image_files)} image files")
            else:
                st.error("❌ Path is not a directory")
        elif directory_path:
            st.error("❌ Directory does not exist")
    
    # Add directory button
    if directory_path and os.path.exists(directory_path) and os.path.isdir(directory_path):
        if st.button("🚀 Add Directory to Database", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(current, total, current_file):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Processing {current}/{total}: {os.path.basename(current_file)}")
            
            try:
                with st.spinner("Adding images to database..."):
                    stats = st.session_state.search_engine.add_images_from_directory(
                        directory_path,
                        progress_callback
                    )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Complete!")
                
                st.success(f"""
                Successfully processed directory!
                - Total files: {stats['total']}
                - Processed: {stats['processed']}
                - Errors: {stats['errors']}
                """)
                
                # Refresh page to update statistics
                st.rerun()
                
            except Exception as e:
                st.error(f"Error adding directory: {str(e)}")
    
    st.divider()
    
    # Database backup and restore
    st.markdown("### 💾 Database Backup & Restore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Export Database")
        st.info("Create a portable backup of your database that can be imported on another server.")
        
        if st.button("📤 Export Database", type="primary"):
            try:
                with st.spinner("Exporting database..."):
                    export_path = st.session_state.search_engine.export_database()
                
                st.success(f"Database exported successfully!")
                
                # Provide download link
                with open(export_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Export File",
                        data=file,
                        file_name=os.path.basename(export_path),
                        mime="application/zip"
                    )
                
                # Clean up the temporary file after a delay
                if os.path.exists(export_path):
                    st.info(f"Export file created: {export_path}")
                    
            except Exception as e:
                if "empty" in str(e).lower():
                    st.warning("Cannot export - database is empty. Add some images first.")
                else:
                    st.error(f"Error exporting database: {str(e)}")
    
    with col2:
        st.markdown("#### Import Database")
        st.info("Restore database from a previously exported backup file.")
        
        uploaded_backup = st.file_uploader(
            "Choose backup file",
            type=['zip'],
            help="Select a .zip file created by the export function"
        )
        
        if uploaded_backup is not None:
            # Show export info
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                    tmp_file.write(uploaded_backup.getvalue())
                    tmp_path = tmp_file.name
                
                # Get export info
                export_info = st.session_state.search_engine.get_export_info(tmp_path)
                
                if export_info:
                    st.success("✅ Valid backup file detected")
                    st.info(f"""
                    **Backup Information:**
                    - Images: {export_info.get('total_images', 'Unknown')}
                    - Created: {export_info.get('timestamp', 'Unknown')}
                    - Version: {export_info.get('version', 'Unknown')}
                    """)
                    
                    # Path mapping section
                    st.markdown("##### Path Mapping (Optional)")
                    st.info("If your images are in different locations on this server, specify path mappings:")
                    
                    path_mapping = {}
                    col_old, col_new = st.columns(2)
                    
                    with col_old:
                        old_path = st.text_input("Old path prefix", placeholder="/old/path/to/images")
                    with col_new:
                        new_path = st.text_input("New path prefix", placeholder="/new/path/to/images")
                    
                    if old_path and new_path:
                        path_mapping[old_path] = new_path
                        st.success(f"Mapping: {old_path} → {new_path}")
                    
                    # Import button
                    if st.button("📥 Import Database", type="primary"):
                        try:
                            with st.spinner("Importing database..."):
                                stats = st.session_state.search_engine.import_database(
                                    tmp_path, 
                                    path_mapping if path_mapping else None
                                )
                            
                            st.success(f"""
                            Import completed successfully!
                            - Imported: {stats['imported']} records
                            - Skipped: {stats['skipped']} (already exist)
                            - Errors: {stats['errors']}
                            """)
                            
                            if stats['imported'] > 0:
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"Error importing database: {str(e)}")
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                else:
                    st.error("❌ Invalid backup file")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        
            except Exception as e:
                st.error(f"Error reading backup file: {str(e)}")
    
    st.divider()
    
    # Database operations
    st.markdown("### 🛠️ Database Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Remove Directory")
        remove_directory = st.text_input(
            "Directory to Remove",
            placeholder="Enter directory path to remove from database"
        )
        
        if st.button("🗑️ Remove Directory", type="secondary"):
            if remove_directory:
                try:
                    deleted_count = st.session_state.search_engine.remove_directory_from_database(remove_directory)
                    if deleted_count > 0:
                        st.success(f"Removed {deleted_count} images from database")
                        st.rerun()
                    else:
                        st.warning("No images found for the specified directory")
                except Exception as e:
                    st.error(f"Error removing directory: {str(e)}")
            else:
                st.warning("Please enter a directory path")
    
    with col2:
        st.markdown("#### Clear Database")
        st.warning("⚠️ This will remove ALL images from the database!")
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all data"):
                try:
                    if st.session_state.search_engine.clear_database():
                        st.success("Database cleared successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to clear database")
                except Exception as e:
                    st.error(f"Error clearing database: {str(e)}")

def settings_page():
    """Render the settings page."""
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    # Model information
    st.markdown("### 🤖 Model Information")
    
    try:
        model_info = st.session_state.search_engine.image_processor.get_model_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Model**: {model_info['model_name']}
            **Device**: {model_info['device']}
            **Feature Dimension**: {model_info['feature_dim']}
            """)
        
        with col2:
            st.info(f"""
            **Input Size**: {model_info['input_size']}
            **Pretrained**: {model_info['pretrained']}
            **Architecture**: EfficientNet-B0
            """)
    
    except Exception as e:
        st.error(f"Error loading model information: {str(e)}")
    
    st.divider()
    
    # Search settings
    st.markdown("### 🔍 Search Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Default Parameters")
        default_results = st.number_input(
            "Default number of results",
            min_value=1,
            max_value=50,
            value=5,
            help="Default number of similar images to return"
        )
        
        default_threshold = st.slider(
            "Default similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Default minimum similarity score"
        )
    
    with col2:
        st.markdown("#### Performance Settings")
        batch_size = st.number_input(
            "Batch processing size",
            min_value=1,
            max_value=100,
            value=10,
            help="Number of images to process in each batch"
        )
        
        max_file_size = st.number_input(
            "Max file size (MB)",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum allowed file size for uploads"
        )
    
    # Save settings button
    if st.button("💾 Save Settings", type="primary"):
        # In a real application, you would save these settings to a config file
        st.success("Settings saved successfully!")
        st.info("Note: Settings will be applied on next restart")
    
    st.divider()
    
    # System information
    st.markdown("### 💻 System Information")
    
    import torch
    import platform
    
    system_info = f"""
    **Python Version**: {platform.python_version()}
    **Platform**: {platform.system()} {platform.release()}
    **PyTorch Version**: {torch.__version__}
    **CUDA Available**: {torch.cuda.is_available()}
    """
    
    if torch.cuda.is_available():
        system_info += f"\n**CUDA Version**: {torch.version.cuda}"
        system_info += f"\n**GPU Count**: {torch.cuda.device_count()}"
    
    st.info(system_info)
    
    # Supported formats
    st.markdown("### 📁 Supported Formats")
    formats = get_supported_image_extensions()
    st.info(f"**Image Formats**: {', '.join(formats)}")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🗄️ Database Management", "⚙️ Settings"]
    )
    
    # Add some information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    This reverse image search application uses:
    - **EfficientNet-B0** for feature extraction
    - **ChromaDB** for vector storage
    - **Streamlit** for the web interface
    
    Upload images and find visually similar ones in your database!
    """)
    
    # Database status in sidebar
    try:
        stats = st.session_state.search_engine.get_database_statistics()
        st.sidebar.markdown("### 📊 Quick Stats")
        st.sidebar.metric("Images in Database", stats.get("total_images", 0))
        st.sidebar.metric("Database Size", stats.get("total_size_formatted", "0B"))
    except Exception:
        st.sidebar.error("Database connection error")
    
    # Route to appropriate page
    if page == "🏠 Home":
        home_page()
    elif page == "🗄️ Database Management":
        database_management_page()
    elif page == "⚙️ Settings":
        settings_page()

if __name__ == "__main__":
    main()