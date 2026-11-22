"""
Rename all occurrences of SentimentIQ to Sentiment Analysis across project.
"""
import os, sys

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
skip_exts = {'.pkl', '.pyc', '.pt', '.csv', '.git', '.cache'}

replacements = [
    ("SentimentIQ –", "Sentiment Analysis –"),
    ("SentimentIQ v2.0", "Sentiment Analysis v2.0"),
    ("SentimentIQ Flask", "Sentiment Analysis Flask"),
    ("Starting SentimentIQ", "Starting Sentiment Analysis"),
    ("SENTIMENTIQ_SECRET", "SENTIMENT_ANALYSIS_SECRET"),
    ("sentimentiq-secret-2024", "sentiment-analysis-secret-key"),
    ("name: sentimentiq-flask", "name: sentiment-analysis-flask"),
    ('raw.get("model_name", "SentimentIQ")', 'raw.get("model_name", "Sentiment Analysis")'),
    ("SentimentIQ", "Sentiment Analysis"),
    ("sentimentiq", "sentiment_analysis"),
]

updated_files = []

for root, dirs, files in os.walk(base):
    if '.git' in dirs: dirs.remove('.git')
    if '__pycache__' in dirs: dirs.remove('__pycache__')
    if '.cache' in dirs: dirs.remove('.cache')
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in skip_exts: continue
        filepath = os.path.join(root, f)
        
        # Don't modify rename_all.py itself
        if os.path.basename(filepath) == "rename_all.py":
            continue
            
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as fp:
            orig = fp.read()
        
        new_content = orig
        lines = new_content.split('\n')
        new_lines = []
        for line in lines:
            # Preserve github remote repo link if exact URL
            if "github.com/karthikmoka/SentimentIQ-Flask" in line:
                new_lines.append(line)
            elif "scratch\\SentimentIQ" in line or "scratch/SentimentIQ" in line:
                new_lines.append(line)
            else:
                updated_line = line
                for old, new in replacements:
                    updated_line = updated_line.replace(old, new)
                new_lines.append(updated_line)
                
        new_content = '\n'.join(new_lines)
        if new_content != orig:
            with open(filepath, 'w', encoding='utf-8') as fp:
                fp.write(new_content)
            updated_files.append(os.path.relpath(filepath, base))

print(f"Successfully updated {len(updated_files)} files:")
for u in updated_files:
    print(f"  ✓ {u}")
