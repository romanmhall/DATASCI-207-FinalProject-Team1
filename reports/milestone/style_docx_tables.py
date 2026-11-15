import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import subprocess, os

def set_cell_background(cell, hex_color):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="CBD5E1", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders_xml = f'''
    <w:tblBorders {nsdecls("w")}>
        <w:top w:val="{val}" w:sz="8" w:space="0" w:color="003262"/>
        <w:left w:val="none"/>
        <w:bottom w:val="{val}" w:sz="8" w:color="003262"/>
        <w:right w:val="none"/>
        <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:insideV w:val="none"/>
    </w:tblBorders>
    '''
    tblPr.append(parse_xml(borders_xml))

def style_milestone_docx():
    base_dir = r"C:\Users\roman\school\DATASCI207\2_Projects\credit_project\reports\milestone"
    md_path = os.path.join(base_dir, "milestone_report.md")
    docx_path = os.path.join(base_dir, "milestone_report.docx")

    # 1. Compile fresh docx from markdown
    subprocess.run(["pandoc", md_path, "-o", docx_path], check=True)
    print("Pandoc compiled docx successfully.")

    doc = docx.Document(docx_path)
    
    # Column width specifications for the 3 tables (in Inches)
    table_widths = {
        0: [Inches(1.5), Inches(0.9), Inches(0.85), Inches(0.95), Inches(2.8)], # Dataset table
        1: [Inches(1.1), Inches(1.8), Inches(2.0), Inches(2.1)],                 # Model table
        2: [Inches(1.2), Inches(1.5), Inches(1.9), Inches(2.4)]                  # Team table
    }

    for t_idx, table in enumerate(doc.tables):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(table, color="CBD5E1", sz="4")

        col_widths = table_widths.get(t_idx, None)

        for r_idx, row in enumerate(table.rows):
            is_header = (r_idx == 0)
            
            # Row height & cant-split
            trPr = row._tr.get_or_add_trPr()
            trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
            if is_header:
                trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

            # Alternating background colors
            if is_header:
                row_bg = "003262" # Berkeley Blue
            elif r_idx % 2 == 1:
                row_bg = "FFFFFF"
            else:
                row_bg = "F8FAFC" # Subtle slate tint

            for c_idx, cell in enumerate(row.cells):
                set_cell_background(cell, row_bg)
                set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

                # Set explicit width
                if col_widths and c_idx < len(col_widths):
                    cell.width = col_widths[c_idx]

                # Style paragraphs and runs in cell
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.space_after = Pt(2)
                    p.paragraph_format.line_spacing = 1.05

                    for run in p.runs:
                        run.font.name = "Calibri"
                        run.font.size = Pt(9 if not is_header else 9.5)
                        if is_header:
                            run.font.bold = True
                            run.font.color.rgb = RGBColor(255, 255, 255)
                        else:
                            run.font.color.rgb = RGBColor(15, 23, 42)

    doc.save(docx_path)
    print(f"SUCCESS: Enhanced publication styling applied to all {len(doc.tables)} tables in:\n{docx_path}")

if __name__ == "__main__":
    style_milestone_docx()
