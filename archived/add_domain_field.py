"""
Add domain field to JSONL dataset files

This script adds a "domain" field to existing JSONL files:
- dev_arxiv.jsonl → adds "domain": "arxiv"
- dev_longform.jsonl → adds "domain": "longform"

Original files are preserved with .backup extension.
"""

import json
import shutil
from pathlib import Path

def add_domain_to_jsonl(input_path: str, domain: str):
    """Add domain field to each record in a JSONL file"""
    
    input_file = Path(input_path)
    backup_file = input_file.with_suffix(input_file.suffix + '.backup')
    temp_file = input_file.with_suffix('.tmp')
    
    # Backup original file
    if not backup_file.exists():
        shutil.copy2(input_file, backup_file)
        print(f"✓ Backed up: {backup_file}")
    
    # Process file
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(temp_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if line.strip():
                record = json.loads(line)
                record['domain'] = domain
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
    
    # Replace original with updated version
    temp_file.replace(input_file)
    print(f"✓ Added domain '{domain}' to {count} records in {input_file.name}")
    
    return count

def main():
    """Add domain field to both dataset files"""
    
    datasets = [
        ("src/data/dev_arxiv.jsonl", "arxiv"),
        ("src/data/dev_longform.jsonl", "longform"),
    ]
    
    total = 0
    for filepath, domain in datasets:
        if Path(filepath).exists():
            count = add_domain_to_jsonl(filepath, domain)
            total += count
        else:
            print(f"⚠ File not found: {filepath}")
    
    print(f"\n✅ Total: {total} records updated across {len(datasets)} files")
    print(f"✅ Backup files created with .backup extension")

if __name__ == "__main__":
    main()
