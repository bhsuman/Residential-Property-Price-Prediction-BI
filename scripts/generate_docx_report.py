"""
Compile docs/technical_report.md into a native Microsoft Word (.docx) file
using Python's standard library (zipfile + xml.sax.saxutils).
Zero third-party dependencies required.
"""

import os
import re
import zipfile
from xml.sax.saxutils import escape

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
MD_PATH = os.path.join(DOCS_DIR, "technical_report.md")
DOCX_PATH = os.path.join(DOCS_DIR, "COMP8811_Assignment2_Technical_Report.docx")
HTML_PATH = os.path.join(DOCS_DIR, "COMP8811_Assignment2_Technical_Report.html")


def escape_xml(text):
    return escape(text, entities={"'": "&apos;", '"': "&quot;"})


def create_docx(paragraphs_and_tables, output_path):
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
        <w:sz w:val="22"/>
        <w:color w:val="2B2B2B"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="160" w:line="276" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:rPr>
      <w:b/>
      <w:sz w:val="32"/>
      <w:color w:val="1E3A8A"/>
    </w:rPr>
    <w:pPr>
      <w:spacing w:before="360" w:after="180"/>
    </w:pPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:rPr>
      <w:b/>
      <w:sz w:val="26"/>
      <w:color w:val="1E3A8A"/>
    </w:rPr>
    <w:pPr>
      <w:spacing w:before="240" w:after="120"/>
    </w:pPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:rPr>
      <w:b/>
      <w:sz w:val="23"/>
      <w:color w:val="334155"/>
    </w:rPr>
    <w:pPr>
      <w:spacing w:before="180" w:after="80"/>
    </w:pPr>
  </w:style>
</w:styles>"""

    body_xml_parts = []

    for item in paragraphs_and_tables:
        itype = item["type"]
        if itype == "title":
            body_xml_parts.append(f"""
            <w:p>
              <w:pPr>
                <w:jc w:val="center"/>
                <w:spacing w:before="360" w:after="200"/>
              </w:pPr>
              <w:r>
                <w:rPr>
                  <w:b/>
                  <w:sz w:val="38"/>
                  <w:color w:val="0F172A"/>
                </w:rPr>
                <w:t>{escape_xml(item['text'])}</w:t>
              </w:r>
            </w:p>
            """)
        elif itype == "h1":
            body_xml_parts.append(f"""
            <w:p>
              <w:pPr>
                <w:pStyle w:val="Heading1"/>
              </w:pPr>
              <w:r>
                <w:rPr><w:b/><w:sz w:val="30"/><w:color w:val="1E3A8A"/></w:rPr>
                <w:t>{escape_xml(item['text'])}</w:t>
              </w:r>
            </w:p>
            """)
        elif itype == "h2":
            body_xml_parts.append(f"""
            <w:p>
              <w:pPr>
                <w:pStyle w:val="Heading2"/>
              </w:pPr>
              <w:r>
                <w:rPr><w:b/><w:sz w:val="25"/><w:color w:val="1E3A8A"/></w:rPr>
                <w:t>{escape_xml(item['text'])}</w:t>
              </w:r>
            </w:p>
            """)
        elif itype == "h3":
            body_xml_parts.append(f"""
            <w:p>
              <w:pPr>
                <w:pStyle w:val="Heading3"/>
              </w:pPr>
              <w:r>
                <w:rPr><w:b/><w:sz w:val="23"/><w:color w:val="334155"/></w:rPr>
                <w:t>{escape_xml(item['text'])}</w:t>
              </w:r>
            </w:p>
            """)
        elif itype == "p":
            text = item["text"]
            runs_xml = []
            parts = re.split(r"(\*\*.*?\*\*)", text)
            for p in parts:
                if p.startswith("**") and p.endswith("**"):
                    inner = p[2:-2]
                    runs_xml.append(f"<w:r><w:rPr><w:b/></w:rPr><w:t xml:space=\"preserve\">{escape_xml(inner)}</w:t></w:r>")
                else:
                    if p:
                        runs_xml.append(f"<w:r><w:t xml:space=\"preserve\">{escape_xml(p)}</w:t></w:r>")
            body_xml_parts.append(f"<w:p><w:pPr><w:spacing w:after=\"140\" w:line=\"276\" w:lineRule=\"auto\"/></w:pPr>{''.join(runs_xml)}</w:p>")
        elif itype == "bullet":
            text = item["text"]
            runs_xml = []
            parts = re.split(r"(\*\*.*?\*\*)", text)
            for p in parts:
                if p.startswith("**") and p.endswith("**"):
                    inner = p[2:-2]
                    runs_xml.append(f"<w:r><w:rPr><w:b/></w:rPr><w:t xml:space=\"preserve\">{escape_xml(inner)}</w:t></w:r>")
                else:
                    if p:
                        runs_xml.append(f"<w:r><w:t xml:space=\"preserve\">{escape_xml(p)}</w:t></w:r>")
            body_xml_parts.append(f"""
            <w:p>
              <w:pPr>
                <w:ind w:left="400" w:hanging="240"/>
                <w:spacing w:after="80"/>
              </w:pPr>
              <w:r><w:t>• </w:t></w:r>
              {''.join(runs_xml)}
            </w:p>
            """)
        elif itype == "table":
            headers = item["headers"]
            rows = item["rows"]
            tbl_xml = ["""
            <w:tbl>
              <w:tblPr>
                <w:tblW w:w="9400" w:type="dxa"/>
                <w:tblBorders>
                  <w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>
                  <w:left w:val="none"/>
                  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="94A3B8"/>
                  <w:right w:val="none"/>
                  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
                  <w:insideV w:val="none"/>
                </w:tblBorders>
                <w:tblCellMar>
                  <w:top w:w="120" w:type="dxa"/>
                  <w:bottom w:w="120" w:type="dxa"/>
                  <w:left w:w="160" w:type="dxa"/>
                  <w:right w:w="160" w:type="dxa"/>
                </w:tblCellMar>
              </w:tblPr>
            """]
            # Header Row
            tbl_xml.append("<w:tr><w:trPr><w:tblHeader/></w:trPr>")
            for h in headers:
                tbl_xml.append(f"""
                <w:tc>
                  <w:tcPr>
                    <w:shd w:val="clear" w:color="auto" w:fill="F1F5F9"/>
                  </w:tcPr>
                  <w:p>
                    <w:pPr><w:spacing w:before="60" w:after="60"/></w:pPr>
                    <w:r><w:rPr><w:b/><w:sz w:val="20"/><w:color w:val="0F172A"/></w:rPr><w:t>{escape_xml(h)}</w:t></w:r>
                  </w:p>
                </w:tc>
                """)
            tbl_xml.append("</w:tr>")
            # Data Rows
            for r in rows:
                tbl_xml.append("<w:tr>")
                for cell in r:
                    tbl_xml.append(f"""
                    <w:tc>
                      <w:p>
                        <w:pPr><w:spacing w:before="50" w:after="50"/></w:pPr>
                        <w:r><w:rPr><w:sz w:val="19"/><w:color w:val="334155"/></w:rPr><w:t>{escape_xml(cell)}</w:t></w:r>
                      </w:p>
                    </w:tc>
                    """)
                tbl_xml.append("</w:tr>")
            tbl_xml.append("</w:tbl>")
            body_xml_parts.append("".join(tbl_xml))

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(body_xml_parts)}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types_xml)
        z.writestr("_rels/.rels", rels_xml)
        z.writestr("word/styles.xml", styles_xml)
        z.writestr("word/document.xml", document_xml)

    print(f"[SUCCESS] Native Word document created at: {output_path} ({os.path.getsize(output_path):,} bytes)")


def parse_markdown_to_structure(md_content):
    lines = md_content.splitlines()
    items = []
    in_table = False
    table_headers = []
    table_rows = []
    in_code_block = False

    for line in lines:
        s = line.strip()
        if s.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if s == "---":
            continue

        # Check Table line
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s[1:-1].split("|")]
            if all(set(c) <= {"-", ":"} for c in cells):
                continue
            if not in_table:
                in_table = True
                table_headers = cells
                table_rows = []
            else:
                table_rows.append(cells)
            continue
        else:
            if in_table:
                in_table = False
                items.append({"type": "table", "headers": table_headers, "rows": table_rows})
                table_headers = []
                table_rows = []

        if not s:
            continue

        if s.startswith("# "):
            items.append({"type": "title", "text": s[2:].strip()})
        elif s.startswith("## "):
            items.append({"type": "h1", "text": s[3:].strip()})
        elif s.startswith("### "):
            items.append({"type": "h2", "text": s[4:].strip()})
        elif s.startswith("#### "):
            items.append({"type": "h3", "text": s[5:].strip()})
        elif s.startswith("- ") or s.startswith("* "):
            items.append({"type": "bullet", "text": s[2:].strip()})
        elif re.match(r"^\d+\.\s", s):
            num_match = re.match(r"^(\d+\.\s)(.*)", s)
            items.append({"type": "bullet", "text": num_match.group(2).strip()})
        else:
            items.append({"type": "p", "text": s})

    if in_table:
        items.append({"type": "table", "headers": table_headers, "rows": table_rows})

    return items


def main():
    if not os.path.exists(MD_PATH):
        print(f"Error: {MD_PATH} not found.")
        return

    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    structured = parse_markdown_to_structure(md_text)
    create_docx(structured, DOCX_PATH)


if __name__ == "__main__":
    main()

