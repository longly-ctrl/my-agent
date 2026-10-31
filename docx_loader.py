from hashlib import sha256            #根据文件字节计算摘要，作为文件指纹
from io import BytesIO                #把内存中的字节包装成可读取、可定位的二进制流
from pathlib import Path              #表示文件路径，读取文件
from zipfile import ZipFile           #读取 DOCX 压缩包中的成员文件
import xml.etree.ElementTree as ET    #解析 XML，并遍历其中的节点

#处理 XML 命名空间
NS = {  #保存“前缀 → 命名空间地址”的映射
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
}
W = "{" + NS["w"] + "}" #W 则把这个地址拼成方便匹配 XML 标签的字符串。

def read_heading_levels(archive: ZipFile) -> dict[str, int]:  # 建立样式 ID 到标题级别的映射。
    levels = {}  # 保存映射，例如当前文档中的 "2" 对应一级标题。

    #archive.namelist()   获取 ZIP 包内所有成员的名称列表。
    if "word/styles.xml" not in archive.namelist():  # 检查文档包是否包含样式定义。
        return levels  # 缺少样式文件时返回空映射，不猜测标题。

    styles = ET.fromstring(archive.read("word/styles.xml"))  # 读取并解析样式 XML。

    for style in styles.findall("w:style", NS):  # 遍历样式定义。
        if style.get(W + "type") != "paragraph":  # 只处理段落样式。
            continue  # 跳过字符、表格等其他类型的样式。

        style_id = style.get(W + "styleId")  # 取得正文段落引用的样式标识。
        outline = style.find("w:pPr/w:outlineLvl", NS)  # 查找样式直接声明的大纲级别。

        if style_id is None or outline is None:  # 检查是否具备识别标题所需的信息。
            continue  # 跳过没有直接声明大纲级别的样式。

        value = int(outline.get(W + "val", "9"))  # 读取大纲值，缺少取值时按无级别处理。
        if 0 <= value <= 8:  # 筛选对应标题一到标题九的有效值。
            levels[style_id] = value + 1  # 转换为从 1 开始的标题级别。

    return levels  # 返回普通字典，关闭文档包后仍可使用。

def load_docx(path: Path) -> dict:  #类型标注 path: Path 不会自动把字符串转换成 Path，应传入：load_docx(Path("knowledge/example.docx"))
    """提取正文文本段落，保留原文件指纹和段落序号。"""
    source = path.resolve()     #解析为绝对路径，并解析路径中的符号链接等因素。

    # 读取同一份字节快照，确保指纹与解析内容对应。
    raw = source.read_bytes()   #把整个文件读为 bytes，不进行文本解码。
    #BytesIO(raw)：raw 本身是字节数据，BytesIO 将它包装成内存中的二进制流，使其具备 read()、seek() 等文件式操作。
    with ZipFile(BytesIO(raw)) as archive:  #ZipFile(BytesIO(raw))：从这份内存流中打开 ZIP 包，默认是读取模式。
        root = ET.fromstring(archive.read("word/document.xml"))     #把 XML 解析成树，返回根节点 root。
    #archive.read("word/document.xml")：读取并解压压缩包内部名为 word/document.xml 的成员，返回 XML 字节。它不会把整个压缩包解压到磁盘。
        heading_levels = read_heading_levels(archive)  # 在文档包关闭前读取标题样式。
    
    body = root.find("w:body", NS) # 在 root 的直接子节点中，查找指定命名空间下的 body 元素。
    if body is None:
        raise ValueError("Word 文档缺少正文节点")

    # 包含表格单元格中的段落；空段落也参与原始编号。
    paragraphs = list(body.iter(W + "p"))
    records = []

    for number, paragraph in enumerate(paragraphs, start=1):
        parts = []
        for node in paragraph.iter():   #paragraph.iter() 会遍历这个段落及其所有后代节点；代码根据节点类型决定提取什么。
            if node.tag == W + "t":
                parts.append(node.text or "")
            elif node.tag == W + "tab":
                parts.append("\t")
            elif node.tag in (W + "br", W + "cr"):
                parts.append("\n")

        text = "".join(parts)   #这里使用空字符串连接，意味着相邻文本节点之间不会自动添加空格
        if text.strip():  # 只保存包含实际文本的段落。
            style_node = paragraph.find("w:pPr/w:pStyle", NS)  # 查找该段落引用的样式。
            style_id = None  # 默认该段落没有显式指定样式。

            if style_node is not None:  # 判断段落是否引用了样式。
                style_id = style_node.get(W + "val")  # 读取样式 ID，用于查询映射。

            records.append({  # 保留原有字段，并补充标题元数据。
                "paragraph": number,  # 保留原始段落编号。
                "text": text,  # 保留原始提取文本。
                "heading_level": heading_levels.get(style_id),  # 标题为 1～9，普通段落为 None。
            })
    return {
        "source": str(source),                  #解析后的绝对路径字符串
        "source_hash": sha256(raw).hexdigest(), #原始 DOCX 字节的 SHA-256 摘要
        "total_paragraphs": len(paragraphs),    #找到的全部段落数，包括被过滤的空白段落
        "paragraphs": records,                  #非空段落记录，每条包含原始序号和正文
    }