"""Export the Electronic Scouts measurement-driven robe draft. Units: mm."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

from reportlab.lib.colors import HexColor, Color, black, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from pattern_geometry import DEFAULT_PARAMS, PARAM_SPECS, generate

ROOT = Path(__file__).resolve().parent
INK = HexColor('#24343b')
BLUE = HexColor('#3e728b')
MUTED = HexColor('#61737a')
PAPER = HexColor('#f7f4ec')
PALE = HexColor('#e5ebea')
GOLD = HexColor('#ac744c')
W, H = A4
FONT, BOLD = 'Helvetica', 'Helvetica-Bold'
for regular, bold in [
    (Path('C:/Windows/Fonts/DejaVuSans.ttf'), Path('C:/Windows/Fonts/DejaVuSans-Bold.ttf')),
    (Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')),
]:
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont('Scout', str(regular)))
        pdfmetrics.registerFont(TTFont('ScoutBold', str(bold)))
        FONT, BOLD = 'Scout', 'ScoutBold'
        break


def bounds(piece):
    pts = piece['outline']
    return max(p[0] for p in pts), max(p[1] for p in pts)


def marks(piece):
    result=[]
    bw,bh=bounds(piece)
    for i,line in enumerate(piece.get('lines',[]),1):
        if line.get('label'):
            # Short codes keep sewing marks legible on narrow components.
            pt=line['points'][0]
            result.append({'code':f'M{i}','label':line['label'],
                           'x':max(4,min(bw-12,pt[0]+4)),
                           'y':max(5,min(bh-4,pt[1]-4))})
    return result


def path(c, points, x, top, scale=1, close=False, fill=0):
    if not points:
        return
    p = c.beginPath()
    p.moveTo(x + points[0][0]*mm*scale, top - points[0][1]*mm*scale)
    for px, py in points[1:]:
        p.lineTo(x+px*mm*scale, top-py*mm*scale)
    if close:
        p.close()
    c.drawPath(p, stroke=1, fill=fill)


def piece_art(c, piece, x, top, scale=1, detail=True, tint=False):
    c.saveState()
    c.setStrokeColor(INK)
    c.setFillColor(PALE if tint else white)
    c.setLineWidth(0.65 if scale >= 0.5 else 0.7)
    path(c, piece['outline'], x, top, scale, close=True, fill=int(tint))
    c.setDash(3, 2)
    c.setStrokeColor(BLUE)
    c.setLineWidth(0.4)
    path(c, piece['seam'], x, top, scale, close=True)
    c.setDash()
    for line in piece.get('lines', []):
        kind = line.get('kind','mark')
        c.setStrokeColor(GOLD if kind in ('fold','hem') else BLUE)
        c.setDash(2,2) if kind in ('fold','hem','reference','construction') else c.setDash()
        path(c, line['points'], x, top, scale)
    c.setDash()
    grain = piece.get('grain')
    if grain:
        c.setStrokeColor(MUTED)
        path(c, grain, x, top, scale)
        for i,j in ((0,1),(1,0)):
            ax,ay=grain[i]; bx,by=grain[j]
            dx,dy=bx-ax,by-ay
            length=math.hypot(dx,dy)
            if length:
                ux,uy=dx/length,dy/length
                arrow=4 if scale>=0.5 else 9
                path(c, [[ax+ux*arrow-uy*arrow/2,ay+uy*arrow+ux*arrow/2],[ax,ay],[ax+ux*arrow+uy*arrow/2,ay+uy*arrow-ux*arrow/2]],x,top,scale)
    if detail:
        c.setFont(FONT, 8 if scale>=0.5 else 5)
        c.setFillColor(INK)
        for label in piece.get('labels',[]):
            c.drawString(x+label['x']*mm*scale,top-label['y']*mm*scale,label['text'])
        c.setFont(BOLD,7)
        c.setFillColor(BLUE)
        for mark in marks(piece):
            c.drawString(x+mark['x']*mm*scale,top-mark['y']*mm*scale,mark['code'])
    c.restoreState()


def svg_piece(piece, params, output):
    bw,bh=bounds(piece)
    margin=15
    sw,sh=max(240,bw+2*margin),bh+2*margin+25
    esc=lambda x: html.escape(str(x),quote=True)
    points=lambda pts: ' '.join(f'{x:.3f},{y:.3f}' for x,y in pts)
    content=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{sw:.3f}mm" height="{sh:.3f}mm" viewBox="0 0 {sw:.3f} {sh:.3f}">',
        f'<title>{esc(piece["id"])} - {esc(piece["name"])}</title>',
        '<desc>Millimeters. 1:1 when physical dimensions are preserved. Prototype. Black: cut; dashed blue: seam or finished boundary. Allowances already included.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="15" y="9" font-family="sans-serif" font-size="4">{esc(piece["id"])} | {esc(piece["name"])} | {esc(piece["cut"])}</text>',
        '<text x="15" y="15" font-family="sans-serif" font-size="3">PROTOTYPE / mm / allowances included / confirm calibration before cutting</text>',
        f'<g transform="translate({margin},{margin+15})">',
        f'<polygon points="{points(piece["outline"])}" fill="none" stroke="#24343b" stroke-width="0.3"/>',
        f'<polygon points="{points(piece["seam"])}" fill="none" stroke="#3e728b" stroke-width="0.2" stroke-dasharray="2 1"/>']
    for line in piece.get('lines',[]):
        dash=' stroke-dasharray="2 1"' if line.get('kind') in ('fold','hem','reference','construction') else ''
        content.append(f'<polyline points="{points(line["points"])}" fill="none" stroke="#ac744c" stroke-width="0.25"{dash}/>')
    if piece.get('grain'):
        content.append(f'<polyline points="{points(piece["grain"])}" fill="none" stroke="#61737a" stroke-width="0.3"/>')
        a,b=piece['grain']; dx,dy=b[0]-a[0],b[1]-a[1]; length=math.hypot(dx,dy)
        if length:
            ux,uy=dx/length,dy/length
            for q,k in ((a,1),(b,-1)):
                tip=[[q[0]+k*ux*4-uy*2,q[1]+k*uy*4+ux*2],q,[q[0]+k*ux*4+uy*2,q[1]+k*uy*4-ux*2]]
                content.append(f'<polyline points="{points(tip)}" fill="none" stroke="#61737a" stroke-width="0.3"/>')
    for label in piece.get('labels',[]):
        content.append(f'<text x="{label["x"]:.3f}" y="{label["y"]:.3f}" font-family="sans-serif" font-size="3">{esc(label["text"])}</text>')
    for mark in marks(piece):
        content.append(f'<text x="{mark["x"]:.3f}" y="{mark["y"]:.3f}" font-family="sans-serif" font-size="2.5" fill="#3e728b">{mark["code"]}</text>')
    content+=['</g>',f'<path d="M 15 {sh-7:.3f} h 100 m -100 -2 v 4 m 100 -4 v 4" fill="none" stroke="black" stroke-width="0.3"/>',f'<text x="15" y="{sh-10:.3f}" font-family="sans-serif" font-size="3">100 mm verification line</text>','</svg>']
    output.write_text('\n'.join(content),encoding='utf-8')


def text(c, s, x, top, size=10, bold=False, color=INK):
    c.setFillColor(color); c.setFont(BOLD if bold else FONT,size)
    c.drawString(x,top,s)


def paragraph(c, s, x, top, width, size=9.5, leading=15, color=INK):
    words=s.split(); lines=[]; current=''
    for word in words:
        trial=(current+' '+word).strip()
        if pdfmetrics.stringWidth(trial,FONT,size)>width and current:
            lines.append(current); current=word
        else:
            current=trial
    if current: lines.append(current)
    for line in lines:
        text(c,line,x,top,size,color=color); top-=leading
    return top


def base(c, page, title, subtitle):
    c.setPageSize(A4)
    c.setFillColor(PAPER); c.rect(0,0,W,H,stroke=0,fill=1)
    text(c,'ELECTRONIC SCOUTS    /    GARMENT LAB 01',40,H-34,8,True,BLUE)
    text(c,title,40,H-79,25,True)
    paragraph(c,subtitle,40,H-104,W-80,9,14,MUTED)
    c.setStrokeColor(PALE); c.line(40,36,W-40,36)
    text(c,'PARAMETRIC WRAP ROBE  /  FITTING PROTOTYPE  /  SEPTEMBER 2026',40,23,6.4,color=MUTED)
    c.setFont(FONT,8); c.drawRightString(W-40,23,f'{page:02}')


def panel(c, title, body, x, top, width, number=None):
    if number:
        text(c,number,x,top,9,True,BLUE); x+=26; width-=26
    text(c,title,x,top,11,True)
    return paragraph(c,body,x,top-21,width,9.3,14)-22


def silhouette(c,x,top,s=1):
    """Flat design schematic, deliberately not a fit simulation."""
    def shape(points,fill,stroke=INK):
        c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(0.8)
        p=c.beginPath(); p.moveTo(x+points[0][0]*s,top-points[0][1]*s)
        for px,py in points[1:]: p.lineTo(x+px*s,top-py*s)
        p.close(); c.drawPath(p,fill=1,stroke=1)
    dark=HexColor('#434b50'); light=HexColor('#626c72'); blue=HexColor('#668c9c')
    shape([(49,53),(20,67),(0,136),(36,147),(48,107),(42,228),(148,228),(142,107),(154,147),(190,136),(170,67),(141,53)],blue)
    shape([(56,53),(29,68),(15,113),(47,126),(56,102),(39,294),(97,306),(146,294),(136,102),(144,125),(176,111),(160,69),(136,53)],dark)
    shape([(73,47),(58,58),(71,95),(128,157),(139,82),(129,53)],light)
    shape([(116,51),(134,62),(120,99),(54,185),(58,120),(78,89)],HexColor('#778187'))
    shape([(72,79),(88,90),(68,288),(47,297),(60,156)],light)
    shape([(113,78),(129,91),(139,286),(118,296),(109,153)],light)
    shape([(55,164),(139,164),(141,188),(54,188)],HexColor('#303a40'))
    shape([(109,168),(133,168),(133,184),(109,184)],HexColor('#80898b'))
    shape([(49,65),(53,20),(70,0),(96,-8),(121,0),(139,21),(144,65),(130,79),(122,26),(110,12),(83,12),(69,30),(65,82)],dark)
    c.setStrokeColor(BLUE); c.setLineWidth(1)
    for yy,label in ((20,'generous hood'),(105,'short outer sleeve'),(175,'contained sash'),(263,'removable tabards')):
        c.line(x+147*s,top-yy*s,x+205*s,top-yy*s)
        text(c,label,x+210*s,top-yy*s-3,8,color=MUTED)


MEASURES=[
    ('chest','Full chest over the clothes to be worn underneath.'),
    ('hip','Fullest seat/hip over underlayers.'),
    ('waist','Natural waist, comfortably measured.'),
    ('shoulder_width','Shoulder point to shoulder point across the back.'),
    ('shoulder_to_waist','High shoulder point beside neck down to waist.'),
    ('neck_circumference','Around base of neck; ease is separate.'),
    ('head_circumference','Around brow and widest back of head.'),
    ('head_height','Vertical shoulder-level-to-crown distance; use a helper.'),
    ('upper_arm','Around fullest upper arm over intended underlayer.'),
    ('hand_circumference','Around hand at widest knuckles, thumb tucked.'),
    ('wrist','Wrist circumference; relevant to long, narrower sleeves.'),
    ('robe_length','High shoulder point to chosen robe hem.'),
    ('tunic_length','High shoulder point to chosen tunic hem.'),
    ('sleeve_length','Shoulder point to outer sleeve edge; NOT sleeve pattern length.'),
    ('tunic_sleeve_length','Shoulder point to inner sleeve edge.')]


def guide_pdf(data, output):
    c=canvas.Canvas(str(output),pagesize=A4)
    c.setTitle('Electronic Scouts | Parametric robe - cutting and sewing guide')
    p=data['params']
    base(c,1,'Wear the curiosity.','A practical first draft of the clothing in the Electronic Scouts concept images.')
    silhouette(c,63,615,1.35)
    y=158
    text(c,'ONE PATTERN FAMILY. YOUR MEASUREMENTS.',40,y,10,True,BLUE)
    paragraph(c,'Charcoal wrap robe, a cord-free hood, blue short-sleeve undertunic and optional tabards. Keep the movement easy and the cloth matte. The device carrier remains a separate design until its real dimensions and weight are known.',40,y-22,W-80,10,16)
    c.showPage()
    base(c,2,'Measure the wearer.','All inputs and outputs use millimeters. Example values demonstrate the draft; they are not a standard child size.')
    y=H-155
    for key,desc in MEASURES:
        if key not in p: continue
        text(c,key,40,y,8.3,True)
        c.setFont(BOLD,9); c.drawRightString(W-40,y,str(p[key])+' mm')
        y=paragraph(c,desc,40,y-13,W-145,8.4,11)-12
    paragraph(c,'Measure over the intended underlayers, then add ease through the separate ease controls. Do not scale every dimension by age or height: heads, hands, shoulders and torso proportions vary independently.',40,86,W-80,8.8,13)
    c.showPage()
    base(c,3,'Set the character.','Edit measurements.example.json, save a wearer-specific copy, then regenerate the entire set.')
    y=H-157
    settings=[('FIT + WRAP',['chest_ease','hip_ease','sleeve_ease','cuff_ease','neck_ease','wrap_overlap','hem_flare']),('HOOD + ACCESSORIES',['hood_ease','sash_width','sash_overlap','sash_ease','tabard_width','tabard_drop']),('FINISH + COMPONENTS',['seam_allowance','hem_allowance','binding_width','include_hood','include_tunic','include_tabards','include_sash'])]
    for title,keys in settings:
        text(c,title,40,y,10,True,BLUE); y-=25
        for key in keys:
            if key not in p:continue
            text(c,key,48,y,9)
            val=p[key]; rendered=('yes' if val else 'no') if isinstance(val,bool) else f'{val} mm'
            c.setFont(BOLD,9); c.drawRightString(280,y,rendered); y-=19
        y-=17
    y=panel(c,'A draft is a hypothesis.','The formulas match adjoining seam lengths and produce a consistent cutting shape. They cannot establish comfort, hood balance, drape or fit on a particular wearer. Make a toile in inexpensive fabric with similar weight before cutting the good cloth.',320,H-156,W-360)
    y=panel(c,'Choose the first silhouette.','The example has short outer sleeves, a slightly longer blue sleeve underneath and an activity-length hem. Length, ease and accessories are independent choices. Long sleeves need a fresh reach and cuff check.',320,y,W-360)
    panel(c,'Finish the fit at the waist.','Use a broad, short fastening sash with hook-and-loop tabs. Position closures during fitting. Keep the hood cord-free; let detachable decoration provide the ceremony.',320,y,W-360)
    c.showPage()
    page=4
    if data.get('warnings'):
        base(c,page,'Fit these details.','Notices calculated for this measurement configuration. Address them on the toile.')
        y=H-160
        for i,notice in enumerate(data['warnings'],1):
            y=panel(c,f'Fitting note {i}',notice,40,y,W-80)
        c.showPage();page+=1
    # Complete per-piece notes beside each diagram, not only a thumbnail index.
    for start in range(0,len(data['pieces']),2):
        base(c,page,'The cuts.', 'Black outlines include allowances. Dashed blue lines show seam or finished boundaries. Diagrams on this page are not to scale.')
        for i,piece in enumerate(data['pieces'][start:start+2]):
            x=40; top=H-155-i*310
            bw,bh=bounds(piece); scale=min(185/(bw*mm),180/(bh*mm))
            text(c,f'{piece["id"]}  /  {piece["name"]}',x,top,9,True)
            paragraph(c,piece['cut'],x,top-17,W-80,8,11,MUTED)
            piece_art(c,piece,x+5,top-42,scale,detail=False,tint=True)
            paragraph(c,f'Cut envelope: {bw:.1f} x {bh:.1f} mm. {piece.get("material","")}',x,top-239,190,8,11)
            yy=top-43
            for note in piece.get('notes',[]):
                yy=paragraph(c,note,250,yy,W-290,8,11)-7
            if marks(piece):
                yy-=3
                text(c,'MARKS ON FULL-SIZE CUTS',250,yy,7,True,BLUE)
                yy-=14
                paragraph(c,' / '.join(m['code']+' '+m['label'] for m in marks(piece)),250,yy,W-290,7.3,10)
        c.showPage(); page+=1
    base(c,page,'Print. Prove. Cut.','Use either the supplied A4 tile set or the variable-page-size full-size PDF. SVGs retain millimeter dimensions.')
    y=H-161
    for i,title,body in [
        ('01','Check the paper scale.','Print the calibration page at Actual size / 100%. Disable Fit, Shrink and printer scaling. The square must measure 50 mm in both directions and the bar must measure 100 mm. Correct scaling before printing the pattern.'),
        ('02','Assemble only what you need.','A4 tiles are grouped by piece ID, row and column. Each blue frame is 190 x 250 mm, with a 10 mm overlap. Match duplicate marks and lines in that overlap; do not butt page edges. Recheck a 100 mm interval after assembly. Cut one paper template per unique shape.'),
        ('03','Use the right line.','The solid black line is the fabric cutting line. Allowances are included: do not add them twice. Dashed blue is the sewing or finished-edge boundary, depending on the edge notes. Transfer notches and grain lines with removable marks.'),
        ('04','Plan the fabric before buying.','Prewash and press. Arrange all copies on the usable fabric width with grain arrows parallel to the selvage. Mirrored pairs require opposite handed pieces. Cut envelopes in the CSV are planning rectangles, not an optimized fabric layout or a guaranteed yardage.'),
        ('05','Prototype the movement.','Baste a toile; fasten it at the planned waist. Reach overhead and across the body, kneel, squat, sit, take a long step and turn with the hood up. Adjust restricted seams, dragging hems, sleeve clearance and hood view, then regenerate.')]:
        y=panel(c,title,body,40,y,W-80,i)
    c.showPage(); page+=1
    base(c,page,'Build in layers.','Press after each operation. Use the exact edge allowances and finishing notes for each piece.')
    y=H-160
    for i,title,body in [
        ('01','Prepare the pieces.','Cut the required copies, marking wrong sides and pairs. Transfer matching marks. Finish raw joining edges by overlock or zigzag as appropriate. Staystitch the slanting neck edges so they do not stretch during handling.'),
        ('02','Join shoulders, then sleeves.','Sew front shoulders to back shoulders, right sides together. Open the body flat. Match each sleeve top midpoint to the shoulder seam and its ends to the lower armhole marks; sew at the joining allowance.'),
        ('03','Close sides and underarms.','Fold the garment right sides together. Sew each sleeve underarm and corresponding body side, pivoting at the armhole corner. Check both layers meet their marks without forcing or gathering.'),
        ('04','Make and attach the hood.','Join the two hood halves on the marked crown/back edges only. Finish the face edge as specified. Match hood center back to robe center back and base ends to neckline endpoints; sew the base to the neck. Clip into angular neck allowance as needed without cutting stitches.'),
        ('05','Finish openings and hems.','Use each piece note for cuffs and lower hems. Join bias strip short ends with straight seams at the joining allowance, press open, then fold lengthwise into four bands and bind the specified edges. Make the undertunic in the same shoulder/sleeve/side order.'),
        ('06','Fit sash and tabards last.','Sew accessory layers right sides together at the joining allowance, leave a turning gap, turn and close it. Place short sash fasteners on the toile at the actual overlap. Fit optional tabards over the shoulders and secure them at the sash or with removable snaps.')]:
        y=panel(c,title,body,40,y,W-80,i)
    c.showPage(); page+=1
    base(c,page,'Check the seams.','Geometric checks compare generated stitching lines. A passing result is not a wear test.')
    y=H-157
    for check in data.get('seam_checks',[]):
        if y<160:
            c.showPage(); page+=1
            base(c,page,'Check the seams.','Continued. Millimeters throughout.'); y=H-157
        name=check['name']; a=check['a_mm']; b=check['b_mm']; tol=check.get('tolerance_mm',0.1)
        good=abs(a-b)<=tol
        text(c,name,40,y,8.8,True)
        text(c,f'{a:.2f} / {b:.2f} mm   |   difference {abs(a-b):.3f} mm   |   {"PASS" if good else "FAIL"}',40,y-17,8.2,color=BLUE if good else GOLD)
        y-=43
    paragraph(c,'Also inspect the paper: paired side seams and shoulder seams, sleeve midpoint marks, matching hood endpoints, a continuous cut outline, mirrored front copies and straight grain. If a new parameter combination gives an implausible shape, change the inputs and recheck the toile.',40,113,W-80,9,14)
    c.showPage(); page+=1
    base(c,page,'A uniform for discovery.','Design notes and sources. The geometry is an original prototype, not a traced commercial pattern.')
    y=H-160
    y=panel(c,'Make it repairable.','Use ordinary woven cloth and visible, accessible seams. Add a small removable mission patch: observe, measure, build, explain. Let repairs and experiments become part of the garment\'s history.',40,y,W-80)
    y=panel(c,'Keep the device separate.','The concept terminal has no reliable physical dimensions in this folder. Design its removable carrier after measuring the real enclosure, controls and weight. The cloth pattern does not yet include a load-bearing harness.',40,y,W-80)
    sources=[('Reference artwork','Young scout in futuristic setting.png and Futuristic scout in a vibrant world.png, in the parent project folder. These inform appearance, not body dimensions.',None),('Construction reference','BERNINA / WeAllSew: Sewing Tutorial - Make an Easy Kimono Top. Flat sleeve assembly, pressing and print calibration.','https://weallsew.com/sewing-tutorial-make-easy-kimono-top/'),('Hood reference','Pattern Studio 101: 2 Piece Hood Pattern Making. Neck drop and front extension affect hood balance.','https://www.patternstudio101.com/blog/2-piece-hood-pattern-making'),('Printing reference','Adobe Acrobat: Print large documents. Poster tile scaling and overlap.','https://helpx.adobe.com/acrobat/desktop/print-documents/set-up-and-print-pdfs/large-documents.html'),('Cord-free hood','US CPSC: Drawstrings in Children\'s Upper Outerwear. Supports choosing snaps or hook-and-loop in place of neck or hood cords.','https://www.cpsc.gov/Business--Manufacturing/Business-Education/FAQ?p=2993&tid%5B3001%5D=3001')]
    for title,body,url in sources:
        text(c,title,40,y,9.5,True,BLUE); y-=20
        if url:
            c.linkURL(url,(40,y+15,W-40,y+30),relative=0)
        y=paragraph(c,body,40,y,W-80,8.8,13)-23
    text(c,'Sources reviewed 5 September 2026. Follow the generated piece notes.',40,64,8,color=MUTED)
    c.save()


def calibration(c,title,subtitle):
    c.setPageSize(A4)
    text(c,'ELECTRONIC SCOUTS / CUTTING PATTERN',30,H-40,10,True,BLUE)
    text(c,title,30,H-80,22,True)
    paragraph(c,subtitle,30,H-107,W-60,10,16)
    c.setStrokeColor(black); c.setLineWidth(0.7)
    c.rect(30,H-310,50*mm,50*mm)
    text(c,'50 x 50 mm',43,H-242,12,True)
    text(c,'Measure BOTH axes',43,H-262,8)
    yy=H-360; c.line(30,yy,30+100*mm,yy)
    for dx in range(0,101,10): c.line(30+dx*mm,yy-4,30+dx*mm,yy+4)
    text(c,'100 mm',30,yy-21,10,True)
    paragraph(c,'Print this page first at 100% / Actual size. Disable Fit and Shrink. Do not proceed until both square edges and the full bar measure correctly. These example cuts use illustrative measurements; regenerate for the wearer before cutting final fabric.',30,H-422,W-60,11,18)


def master_pdf(data,output):
    c=canvas.Canvas(str(output),pagesize=A4)
    c.setTitle('Electronic Scouts | 1:1 pattern master (variable page sizes)')
    calibration(c,'Full-size master','Page 1 is A4. Remaining pages use custom sizes, one unique pattern piece per page. Print on a plotter or use Adobe Poster at 100% with overlap.'); c.showPage()
    for piece in data['pieces']:
        bw,bh=bounds(piece); pw=max(160,bw+30)*mm; ph=(bh+65)*mm
        c.setPageSize((pw,ph))
        text(c,f'{piece["id"]} / {piece["name"]}',15*mm,ph-13*mm,12,True)
        text(c,piece['cut']+' | 1:1 | mm | allowances included',15*mm,ph-21*mm,8)
        piece_art(c,piece,15*mm,ph-30*mm)
        text(c,'PROTOTYPE - verify wearer measurements and make a toile',15*mm,18*mm,7)
        c.setStrokeColor(black); c.line(15*mm,9*mm,115*mm,9*mm)
        c.line(15*mm,7*mm,15*mm,11*mm); c.line(115*mm,7*mm,115*mm,11*mm)
        text(c,'100 mm',15*mm,12*mm,7)
        c.showPage()
    c.save()


def tile_plan(data):
    plans=[]; page=2
    for piece in data['pieces']:
        bw,bh=bounds(piece)
        cols=max(1,math.ceil((bw+10-10)/180)); rows=max(1,math.ceil((bh+10-10)/240))
        count=cols*rows
        plans.append({'id':piece['id'],'name':piece['name'],'columns':cols,'rows':rows,'first_page':page,'last_page':page+count-1})
        page+=count
    return plans


def tiled_pdf(data,output):
    c=canvas.Canvas(str(output),pagesize=A4)
    c.setTitle('Electronic Scouts | A4 1:1 pattern tiles')
    plans=tile_plan(data)
    calibration(c,'A4 tiled cuts','Print at 100% / Actual size. Each blue frame is 190 x 250 mm. Neighboring tiles overlap by 10 mm. Match repeated lines or crosses; page edges do not butt together.')
    yy=H-535
    for plan in plans:
        if yy<45:
            raise ValueError('Tile index exceeds calibration page')
        text(c,f'{plan["id"]}: pages {plan["first_page"]}-{plan["last_page"]}  |  {plan["columns"]} columns x {plan["rows"]} rows',30,yy,8)
        yy-=17
    c.showPage()
    page=2
    for piece,plan in zip(data['pieces'],plans):
        for row in range(plan['rows']):
            for col in range(plan['columns']):
                c.setPageSize(A4)
                text(c,f'{piece["id"]} / {piece["name"]}',10*mm,H-10*mm,10,True)
                text(c,f'ROW {row+1}/{plan["rows"]}   COLUMN {col+1}/{plan["columns"]}   |   {piece["cut"]}',10*mm,H-16*mm,7)
                x0=10*mm; top=H-25*mm; bottom=top-250*mm
                c.saveState()
                clip=c.beginPath();clip.rect(x0,bottom,190*mm,250*mm);c.clipPath(clip,stroke=0)
                ox=col*180; oy=row*240
                c.setStrokeColor(HexColor('#b5c7ce'));c.setLineWidth(0.3)
                # Grid crosses have identical world coordinates on neighboring pages.
                grid_x=set(range(0,math.ceil((ox+190)/50)*50+1,50))
                grid_y=set(range(0,math.ceil((oy+250)/50)*50+1,50))
                # Every overlap has shared marks, even when no cut edge crosses it.
                grid_x.update(185+180*k for k in range(plan['columns']-1))
                grid_y.update(245+240*k for k in range(plan['rows']-1))
                for gx in sorted(grid_x):
                    for gy in sorted(grid_y):
                        xx=x0+(gx-ox)*mm; yy=top-(gy-oy)*mm
                        c.line(xx-2*mm,yy,xx+2*mm,yy);c.line(xx,yy-2*mm,xx,yy+2*mm)
                piece_art(c,piece,x0+(5-ox)*mm,top-(5-oy)*mm)
                c.restoreState()
                c.setStrokeColor(BLUE);c.setLineWidth(0.4);c.rect(x0,bottom,190*mm,250*mm)
                # Indicate duplicate strip at right and bottom.
                c.setDash(2,2)
                if col<plan['columns']-1:c.line(x0+180*mm,bottom,x0+180*mm,top)
                if row<plan['rows']-1:c.line(x0,top-240*mm,x0+190*mm,top-240*mm)
                c.setDash()
                text(c,f'100% / mm / overlap 10 mm / world origin {ox}, {oy} mm / page {page}',10*mm,12*mm,7)
                c.showPage();page+=1
    c.save()
    return plans


def export(data, output, make_tiles=True):
    pdf=output/'pdf'; svg=output/'svg'; pdf.mkdir(parents=True,exist_ok=True);svg.mkdir(parents=True,exist_ok=True)
    if not make_tiles and (pdf/'scout-robe-a4.pdf').exists():
        raise ValueError('--no-tiles would leave an existing A4 set with stale measurements. Use a new --output folder or regenerate with tiles.')
    for piece in data['pieces']:
        svg_piece(piece,data['params'],svg/f'{piece["id"]}.svg')
    guide_pdf(data,pdf/'scout-robe-guide.pdf')
    master_pdf(data,pdf/'scout-robe-full-size.pdf')
    if make_tiles:data['tile_plan']=tiled_pdf(data,pdf/'scout-robe-a4.pdf')
    (output/'pattern.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    params_doc=['# Measurement and option reference','', 'All dimensions in millimeters. Defaults are illustrative. Use a wearer-specific JSON file.','', '| Key | Current value | Meaning |', '| --- | --- | --- |']
    for key,spec in PARAM_SPECS.items():
        params_doc.append(f'| `{key}` | {data["params"][key]} | {spec["description"]} |')
    params_doc+=['','## Formulas','']+['- '+n for n in data.get('formula_notes',[])]+['','## Parameter notices','']+(['- '+n for n in data.get('warnings',[])] or ['No notices for this configuration.'])
    (output/'parameter-reference.md').write_text('\n'.join(params_doc)+'\n',encoding='utf-8')
    with (output/'cut-list.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['piece','name','group','cut','material','envelope_width_mm','envelope_height_mm'])
        for piece in data['pieces']:
            bw,bh=bounds(piece);writer.writerow([piece['id'],piece['name'],piece.get('group',''),piece['cut'],piece.get('material',''),round(bw,2),round(bh,2)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements',type=Path,help='JSON with measurement/option overrides; unknown keys fail')
    parser.add_argument('--output',type=Path,default=ROOT/'output')
    parser.add_argument('--no-tiles',action='store_true')
    parser.add_argument('--write-example',type=Path,help='Write all default inputs and exit')
    args=parser.parse_args()
    if args.write_example:
        args.write_example.write_text(json.dumps(DEFAULT_PARAMS,indent=2)+'\n',encoding='utf-8');return
    params={}
    if args.measurements:
        params=json.loads(args.measurements.read_text(encoding='utf-8-sig'))
    data=generate(params)
    # Normalized schema also stores every resolved input, not only overrides.
    data.setdefault('params',{**DEFAULT_PARAMS,**params})
    for check in data.get('seam_checks',[]):
        if abs(check['a_mm']-check['b_mm'])>check.get('tolerance_mm',0.1):
            raise ValueError(f'Seam mismatch: {check}')
    export(data,args.output.resolve(),not args.no_tiles)
    print(json.dumps({'output':str(args.output.resolve()),'pieces':len(data['pieces']),'seam_checks':len(data.get('seam_checks',[])),'warnings':data.get('warnings',[]),'a4_pages':1+sum(t['rows']*t['columns'] for t in data.get('tile_plan',[])) if not args.no_tiles else None},indent=2))


if __name__=='__main__':
    main()
