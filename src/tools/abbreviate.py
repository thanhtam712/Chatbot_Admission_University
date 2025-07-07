def get_abrreviate():
    """
    Get the abbreviation for each major.
    """
    
    abrreviate = {
            "Khoa học máy tính": ["KHMT", "KH MT", "khmt", "kh mt"],
            "Khoa học tài năng": ["KHTN", "KH TN", "khtn", "kh tn"],
            "An toàn tài năng": ["ATTN", "AT TN", "attn", "at tn"],
            "Kỹ thuật máy tính": ["KTMT", "KT MT", "ktmt", "kt mt"],
            "Hệ thống thông tin": ["HTTT", "HT TT", "httt", "ht tt"],
            "Trí tuệ nhân tạo": ["TTNT", "TT NT", "ttnt", "tt nt", "AI"],
            "An toàn thông tin": ["ATTT", "AT TT", "attt", "at tt"],
            "Công nghệ thông tin": ["CNTT", "CN TT", "cntt", "cn tt"],
            "Khoa học dữ liệu": ["DS", "KHDL", "khdl", "ds"],
            "Chương trình tiên tiến Hệ thống thông tin": ["CTTT HTTT", "cttt httt", "cttt ht tt", "CTTT HT TT"],
            "Mạng máy tính và An toàn thông tin": ["MMT & ATTT", "MMT & AT TT", "mmt & attt", "mmt & at tt"],
            "Công nghệ phần mềm": ["SE", "CNTT", "se", "cntt"],
            "Kỹ thuật phần mềm": ["SE", "KTPM", "se", "ktpm", "kt pm", "KT PM"],
            "Mạng máy tính và Truyền thông dữ liệu": ["MMT & TTDL", "MMT & TT DL", "mmt & ttdl", "mmt & tt dl"],
            "Thương mại điện tử": ["EC", "TMDT", "ec", "tmdt"],
            "Truyền thông đa phương tiện": ["MC", "TTDPT", "mc", "ttdpt"],
            "Thiết kế vi mạch": ["TKVM", "TK VM", "tkvm", "tk vm", "VLSI", "vlsi", "IC design", "ic design"],
            "Thông tin": ["TT", "tt"],
            "Máy tính": ["MT", "mt"],
            "Chương trình tiên tiến": ["CTTT", "cttt"],
            "Birmingham City University (BCU)": ["BCU", "bcu"],
            "Chương trình đặc biệt": ["OEP", "oep"],
            "Chất lượng cao": ["CLC", "clc"],
            "Văn bằng 2": ["VB2", "vb2"],
            "Liên thông đại học": ["LTDH", "ltdh"],
            "Chương trình chuẩn": ["đại trà", "chính quy", "chương trình đại trà", "chương trình chính quy"],
            "điểm tuyển sinh": ["điểm tuyển sinh", "điểm tuyển", "điểm chuẩn"],
            "Đánh giá năng lực (đgnl)": ["ĐGNL", "dgnl", "đg nl", "đg năng lực", "đánh giá năng lực"],
        #     "các ngành": ["Công nghệ Thông tin", "Hệ thống Thông tin", "Khoa học Máy tính", "Kỹ thuật Phần mềm", "Kỹ thuật Máy tính", "Mạng máy tính & Truyền thông Dữ liệu", "An toàn Thông tin", "Thương mại Điện tử", "Khoa học Dữ liệu", "Trí tuệ Nhân tạo", "Thiết kế Vi mạch", "Truyền thông Đa phương tiện"],
            "Công nghệ Thông tin, Hệ thống Thông tin, Khoa học Máy tính, Kỹ thuật Phần mềm, Kỹ thuật Máy tính, Mạng máy tính & Truyền thông Dữ liệu, An toàn Thông tin, Thương mại Điện tử, Khoa học Dữ liệu, Trí tuệ Nhân tạo, Thiết kế Vi mạch, Truyền thông Đa phương tiện": ["các ngành"],
            "Chương trình tài năng": ["CTTN", "cttn"],
            "Đại học quốc gia": ["ĐHQG", "đhqg"],
            "Kí túc xá": ["KTX", "ktx"],
    }
    
    return abrreviate

def get_file_img(title_file_csv: str) -> str:
        """
        Get the file image for each file csv.
        """
        
        file_img = {
                "2022_DGNL.csv": "dgnl_2022.jpg",
                "2022_THPT.csv": "thpt_2022.jpeg",
                "2023_DGNL.csv": "dgnl_2023.jpg",
                "2023_THPT.csv": "thpt_2023.png",
                "2024_DGNL.csv": "dgnl_2024.png",
                "2024_THPT.csv": "thpt_2024.png",
                "2020_DGNL.csv": "diem-chuan-2020-2024-sheet1.jpg",
                "2020_THPT.csv": "diem-chuan-2020-2024-sheet1.jpg",
                "2021_DGNL.csv": "diem-chuan-2020-2024-sheet1.jpg",
                "2021_THPT.csv": "diem-chuan-2020-2024-sheet1.jpg",
                "diem-chuan-2020-2024.csv": "diem-chuan-2020-2024-sheet1.jpg",
        }
        
        title_file_img = file_img[title_file_csv]
        
        return title_file_img
