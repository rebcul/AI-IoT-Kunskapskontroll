README.md

# SafeWatch

SafeWatch är ett AI-baserat säkerhetssystem byggt i Python med Streamlit och YOLO.

Projektet analyserar bild, video och livekamera för att upptäcka:
- farliga föremål
- person nära farligt föremål
- kvarlämnade objekt

## Teknik
Projektet använder:
- Python
- Streamlit
- YOLO
- OpenCV

## Hur lösningen fungerar
Två separata modeller används:
- weapon model för farliga föremål
- item model för väskor och liknande objekt

Systemet använder också tracking och regelbaserad logik för att upptäcka misstänkta situationer, till exempel en kvarlämnad väska.

## Starta projektet

Installera bibliotek:
```bash
pip install -r requirements.txt