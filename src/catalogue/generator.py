"""
AXUM ROVER — Catalogue Generator
==================================
Generates PDF catalogue entries and JSON records for each scanned artefact.

Each object gets:
  - One page in the PDF catalogue (photo + classification + inscription + 3D info)
  - One JSON record in the mission file

The PDF is what judges see on the display during the demo.
It should look professional — like a real museum catalogue.

Author: Axum Rover Team
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from loguru import logger

# reportlab for PDF generation
# pip install reportlab (already in requirements)
from reportlab.lib.pagesizes import A4
from reportlab.lib.units     import mm
from reportlab.lib.colors    import HexColor, black, white
from reportlab.pdfgen        import canvas
from reportlab.lib.utils     import ImageReader


# ── Publication palette ───────────────────────────────────────
# The catalogue is a printable conservation report, not a dashboard view.
COLOR_GOLD    = black
COLOR_DARK    = white
COLOR_MID     = HexColor('#F2F2F2')
COLOR_ACCENT  = black
COLOR_SUCCESS = black
COLOR_TEXT    = black
COLOR_SUBTEXT = HexColor('#555555')


class CatalogueGenerator:
    """
    Generates the artefact catalogue — both PDF and JSON formats.

    PDF layout per object (A4 page):
    ┌─────────────────────────────────────────┐
    │  AXUM ROVER  •  Artefact Catalogue      │  ← header (dark bg)
    │  Mission ID: AXUM_20260601_143022        │
    ├────────────────┬────────────────────────┤
    │                │  Object #001           │
    │   PHOTO        │  Type: Pottery         │
    │   (largest     │  Confidence: 91%       │
    │    scan image) │  Period: Aksumite      │
    │                │  Dimensions: 82×64mm   │
    ├────────────────┴────────────────────────┤
    │  INSCRIPTION                            │
    │  Ge'ez text: ሰላም                       │
    │  Translation: "Peace / Greeting"        │
    │  Database match: ETH_001 (similarity 94%)│
    ├─────────────────────────────────────────┤
    │  3D MODEL                               │
    │  Photos: 36  •  Build time: 4m 12s      │
    │  File: AXUM_OBJ001/mesh/textured.obj    │
    ├─────────────────────────────────────────┤
    │  [NEW DISCOVERY badge if applicable]    │
    └─────────────────────────────────────────┘
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.entries    = []  # list of ObjectRecord dicts
        self.pdf_path   = None

    def add_entry(self, record) -> str:
        """
        Add one artefact to the catalogue.
        Immediately generates/appends the PDF page and saves JSON.

        Args:
            record: ObjectRecord dataclass instance

        Returns:
            Path to the entry JSON file
        """
        record_dict = asdict(record) if hasattr(record, '__dataclass_fields__') \
                      else record
        self.entries.append(record_dict)

        # Save individual JSON entry
        json_path = self.output_dir / f"{record_dict['object_id']}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(record_dict, f, indent=2, ensure_ascii=False)

        # Generate/update PDF
        self._regenerate_pdf()

        logger.info(f"Catalogue entry added: {record_dict['object_id']}")
        return str(json_path)

    def generate_mission_summary(self, mission) -> str:
        """
        Generate final mission summary page appended to catalogue PDF.

        Args:
            mission: MissionRecord dataclass instance

        Returns:
            Path to catalogue PDF
        """
        mission_dict = asdict(mission) if hasattr(mission, '__dataclass_fields__') \
                       else mission
        self._regenerate_pdf(mission_summary=mission_dict)
        return str(self.pdf_path)

    def _regenerate_pdf(self, mission_summary: dict = None):
        """Rebuild the full PDF with all current entries."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pdf_path = self.output_dir / "axum_catalogue.pdf"

        c = canvas.Canvas(str(self.pdf_path), pagesize=A4)
        W, H = A4  # 210mm × 297mm

        for entry in self.entries:
            self._draw_artefact_page(c, entry, W, H)
            c.showPage()

        if mission_summary:
            self._draw_summary_page(c, mission_summary, W, H)
            c.showPage()

        c.save()
        logger.debug(f"PDF updated: {self.pdf_path} ({len(self.entries)} entries)")

    def _draw_artefact_page(self, c, entry: dict, W: float, H: float):
        """Draw one artefact catalogue page."""

        # ── Background ────────────────────────────────────────
        c.setFillColor(COLOR_DARK)
        c.rect(0, 0, W, H, fill=1, stroke=0)

        # ── Header bar ────────────────────────────────────────
        header_h = 22 * mm
        c.setFillColor(COLOR_MID)
        c.rect(0, H - header_h, W, header_h, fill=1, stroke=0)

        # Publication rule below the running header
        c.setFillColor(COLOR_GOLD)
        c.rect(12*mm, H - header_h - 1*mm, W - 24*mm, 0.5*mm, fill=1, stroke=0)

        # Header text
        c.setFillColor(COLOR_GOLD)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(12*mm, H - 14*mm, "AXUM ROVER")

        c.setFillColor(COLOR_SUBTEXT)
        c.setFont("Helvetica", 9)
        c.drawString(60*mm, H - 14*mm, "Artefact Digitization & Heritage Preservation System")

        # Object number (top right)
        obj_num = entry.get('sequence_number', 0)
        c.setFillColor(COLOR_GOLD)
        c.setFont("Helvetica-Bold", 11)
        c.drawRightString(W - 12*mm, H - 14*mm, f"#{obj_num:03d}")

        # ── Main content area ─────────────────────────────────
        content_top = H - header_h - 5*mm

        # Left column: photo
        photo_x = 12*mm
        photo_y = content_top - 80*mm
        photo_w = 80*mm
        photo_h = 75*mm

        self._draw_photo(c, entry, photo_x, photo_y, photo_w, photo_h)

        # Right column: classification info
        info_x = photo_x + photo_w + 8*mm
        info_y = content_top - 8*mm
        info_w = W - info_x - 12*mm

        self._draw_classification_info(c, entry, info_x, info_y, info_w)

        # ── Inscription section ───────────────────────────────
        section_y = content_top - 88*mm
        self._draw_inscription_section(c, entry, 12*mm, section_y, W - 24*mm)

        # ── Fragment group section ────────────────────────────
        if entry.get('group_id'):
            group_y = section_y - 48*mm
            self._draw_fragment_group_section(c, entry, 12*mm, group_y, W - 24*mm)
            model_y = group_y - 32*mm
        else:
            model_y = section_y - 48*mm

        # ── 3D model section ──────────────────────────────────
        self._draw_3d_section(c, entry, 12*mm, model_y, W - 24*mm)

        # ── New discovery badge ───────────────────────────────
        if entry.get('is_new_discovery'):
            self._draw_discovery_badge(c, W, H)

        # ── Timestamp footer ──────────────────────────────────
        c.setFillColor(COLOR_SUBTEXT)
        c.setFont("Helvetica", 7)
        ts = entry.get('timestamp', '')[:19].replace('T', ' ')
        c.drawString(12*mm, 8*mm, f"Catalogued: {ts}")
        c.drawRightString(W - 12*mm, 8*mm,
                          f"ID: {entry.get('object_id', 'UNKNOWN')}")

    def _draw_photo(self, c, entry, x, y, w, h):
        """Draw the artefact photo or placeholder."""
        # Photo border
        c.setStrokeColor(COLOR_GOLD)
        c.setLineWidth(0.5)
        c.rect(x, y, w, h, fill=0, stroke=1)

        # Try to load actual photo
        photos = entry.get('photo_paths', [])
        if photos:
            best_idx   = len(photos) // 2
            photo_path = photos[best_idx]
            try:
                img = ImageReader(photo_path)
                c.drawImage(img, x + 1*mm, y + 1*mm,
                            width=w - 2*mm, height=h - 2*mm,
                            preserveAspectRatio=True,
                            anchor='c')
                return
            except Exception:
                pass  # fall through to placeholder

        # Placeholder
        c.setFillColor(COLOR_MID)
        c.rect(x + 1*mm, y + 1*mm, w - 2*mm, h - 2*mm, fill=1, stroke=0)
        c.setFillColor(COLOR_SUBTEXT)
        c.setFont("Helvetica", 9)
        c.drawCentredString(x + w/2, y + h/2, "No photo available")

    def _draw_classification_info(self, c, entry, x, y, w):
        """Draw classification details in right column."""
        line_h = 7*mm

        def row(label, value, value_color=COLOR_TEXT, bold=False):
            nonlocal y
            c.setFillColor(COLOR_SUBTEXT)
            c.setFont("Helvetica", 8)
            c.drawString(x, y, label)

            c.setFillColor(value_color)
            font = "Helvetica-Bold" if bold else "Helvetica"
            c.setFont(font, 9)
            c.drawString(x + 28*mm, y, str(value))
            y -= line_h

        # Object type
        class_name = entry.get('class_name', 'unknown').replace('_', ' ').title()
        conf       = entry.get('class_confidence', 0)
        row("Type:", class_name, COLOR_GOLD, bold=True)
        row("Confidence:", f"{conf:.0%}",
            COLOR_SUCCESS if conf > 0.7 else COLOR_ACCENT)

        y -= 3*mm  # small gap

        # Dimensions
        w_mm = entry.get('width_mm', 0)
        h_mm = entry.get('height_mm', 0)
        d_mm = entry.get('depth_mm', 0)
        dims = f"{w_mm:.0f} × {h_mm:.0f} × {d_mm:.0f} mm" \
               if any([w_mm, h_mm, d_mm]) else "Not measured"
        row("Dimensions:", dims)

        # Source
        source = entry.get('class_source', '')
        row("Detected by:", source.replace('_', ' ').title())

        y -= 3*mm

        # Errors (if any)
        errors = entry.get('errors', [])
        if errors:
            c.setFillColor(COLOR_ACCENT)
            c.setFont("Helvetica", 7)
            c.drawString(x, y, f"Issues: {', '.join(errors[:2])}")
            y -= line_h

    def _draw_inscription_section(self, c, entry, x, y, w):
        """Draw Ge'ez inscription section."""
        section_h = 42*mm

        # Section background
        c.setFillColor(COLOR_MID)
        c.rect(x, y - section_h, w, section_h, fill=1, stroke=0)

        # Gold left accent bar
        c.setFillColor(COLOR_GOLD)
        c.rect(x, y - section_h, 2*mm, section_h, fill=1, stroke=0)

        # Section title
        c.setFillColor(COLOR_GOLD)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x + 5*mm, y - 8*mm, "INSCRIPTION ANALYSIS")

        text = entry.get('inscription_text', '')
        if text:
            # Ge'ez text (large)
            c.setFillColor(COLOR_TEXT)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(x + 5*mm, y - 18*mm, text[:30])

            # Translation
            translation = entry.get('translation_en', '')
            if translation:
                c.setFillColor(COLOR_SUBTEXT)
                c.setFont("Helvetica-Oblique", 9)
                c.drawString(x + 5*mm, y - 26*mm,
                             f'"{translation[:60]}"')

            # Database match
            db_id = entry.get('db_match_id', '')
            if db_id:
                c.setFillColor(COLOR_SUCCESS)
                c.setFont("Helvetica", 8)
                c.drawString(x + 5*mm, y - 34*mm,
                             f"✓ Database match: {db_id}")
            elif entry.get('is_new_discovery'):
                c.setFillColor(COLOR_ACCENT)
                c.setFont("Helvetica-Bold", 8)
                c.drawString(x + 5*mm, y - 34*mm,
                             "★ No database match — potential new discovery")
        else:
            c.setFillColor(COLOR_SUBTEXT)
            c.setFont("Helvetica", 9)
            c.drawString(x + 5*mm, y - 20*mm,
                         "No Ge'ez inscription detected on this artefact")

    def _draw_fragment_group_section(self, c, entry, x, y, w):
        """Draw fragment group membership and pairwise match scores."""
        section_h = 28*mm

        c.setFillColor(COLOR_MID)
        c.rect(x, y - section_h, w, section_h, fill=1, stroke=0)

        c.setFillColor(COLOR_GOLD)
        c.rect(x, y - section_h, 2*mm, section_h, fill=1, stroke=0)

        c.setFillColor(COLOR_GOLD)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x + 5*mm, y - 8*mm, "FRAGMENT GROUP")

        group_id = entry.get('group_id', '')
        group_conf = entry.get('group_conf', 0)
        role = entry.get('group_role', 'ungrouped').upper()

        c.setFillColor(COLOR_TEXT)
        c.setFont("Helvetica", 8)
        c.drawString(x + 5*mm, y - 16*mm,
                     f"{group_id}  |  {role}  |  {group_conf:.0%} reconstruction")

        match_scores = entry.get('match_scores', {})
        if match_scores:
            top = sorted(match_scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
            score_text = "  ".join(f"vs {oid.split('-')[-1]}: {s:.0%}" for oid, s in top)
            c.setFillColor(COLOR_SUBTEXT)
            c.setFont("Helvetica", 7)
            c.drawString(x + 5*mm, y - 22*mm, score_text[:70])

    def _draw_3d_section(self, c, entry, x, y, w):
        """Draw 3D model info section."""
        section_h = 28*mm

        c.setFillColor(COLOR_MID)
        c.rect(x, y - section_h, w, section_h, fill=1, stroke=0)

        c.setFillColor(COLOR_GOLD)
        c.rect(x, y - section_h, 2*mm, section_h, fill=1, stroke=0)

        c.setFillColor(COLOR_GOLD)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x + 5*mm, y - 8*mm, "3D RECONSTRUCTION")

        photo_count  = entry.get('photo_count', 0)
        mesh_path    = entry.get('mesh_path', '')
        mesh_duration = entry.get('mesh_duration', 0)

        c.setFillColor(COLOR_TEXT)
        c.setFont("Helvetica", 8)
        c.drawString(x + 5*mm, y - 16*mm,
                     f"Scan photos: {photo_count} / 36  •  "
                     f"Build time: {mesh_duration:.0f}s")

        if mesh_path:
            c.setFillColor(COLOR_SUCCESS)
            c.setFont("Helvetica", 8)
            short_path = Path(mesh_path).name if mesh_path else ""
            c.drawString(x + 5*mm, y - 22*mm,
                         f"✓ Model: {short_path}")
        else:
            c.setFillColor(COLOR_SUBTEXT)
            c.setFont("Helvetica", 8)
            c.drawString(x + 5*mm, y - 22*mm,
                         "3D model not available")

    def _draw_discovery_badge(self, c, W, H):
        """Draw a prominent NEW DISCOVERY badge."""
        badge_w = 50*mm
        badge_h = 12*mm
        badge_x = W - badge_w - 12*mm
        badge_y = 20*mm

        c.setFillColor(COLOR_ACCENT)
        c.rect(badge_x, badge_y, badge_w, badge_h, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(badge_x + badge_w/2,
                            badge_y + 3.5*mm,
                            "★ NEW DISCOVERY")

    def _draw_summary_page(self, c, mission: dict, W, H):
        """Draw mission summary page (last page of catalogue)."""
        # White paper background
        c.setFillColor(COLOR_DARK)
        c.rect(0, 0, W, H, fill=1, stroke=0)

        # Gray report header
        c.setFillColor(COLOR_MID)
        c.rect(0, H - 30*mm, W, 30*mm, fill=1, stroke=0)
        c.setFillColor(COLOR_TEXT)
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(W/2, H - 18*mm, "MISSION SUMMARY")

        # Stats
        y = H - 50*mm
        stats = [
            ("Mission ID",    mission.get('mission_id', '')),
            ("Start time",    mission.get('start_time', '')[:19].replace('T', ' ')),
            ("End time",      mission.get('end_time', '')[:19].replace('T', ' ')),
            ("Objects found", str(mission.get('total_objects', 0))),
            ("Catalogued",    str(mission.get('completed_objects', 0))),
        ]

        for label, value in stats:
            c.setFillColor(COLOR_SUBTEXT)
            c.setFont("Helvetica", 10)
            c.drawString(30*mm, y, label + ":")
            c.setFillColor(COLOR_GOLD)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(90*mm, y, value)
            y -= 10*mm

        # Discoveries
        objects = mission.get('objects', [])
        discoveries = [o for o in objects if o.get('is_new_discovery')]
        if discoveries:
            y -= 5*mm
            c.setFillColor(COLOR_ACCENT)
            c.setFont("Helvetica-Bold", 11)
            c.drawString(30*mm, y, f"★ {len(discoveries)} new discovery(s) flagged for expert review")