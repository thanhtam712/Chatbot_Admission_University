from src.tools import get_file_img

def get_title_private(title: str) -> str:
    """
    Get the title of the private document and return to the public document.
    """
    
    if title in ["CAM_NANG_UIT.pdf"]:
        title_public = "https://tuyensinh.uit.edu.vn/cam-nang-tuyen-sinh-2024"
    elif title.endswith(".csv"):
        title_public = get_file_img(title)
    elif title in ["output.txt"]:
        title_public = "https://cdn.thuvienphapluat.vn/uploads/Hoidapphapluat/2024/NTKL/21112024/de-an-tuyen-sinh-truong-uit.pdf" 
    else:
        title_public = title
    
    return title_public
