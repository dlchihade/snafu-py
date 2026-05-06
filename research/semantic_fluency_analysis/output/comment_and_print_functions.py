#!/usr/bin/env python3
"""
Comprehensive script to:
1. Add comments/docstrings to all functions in notebook
2. Print all functions to a summary file
"""

import json
import re
import argparse
import sys
from typing import List, Dict
import os
import time
from pathlib import Path
# #region agent log
log_path = Path("/Users/diettachihade/snafu-py/research/semantic_fluency_analysis/output/.cursor/debug.log")
def _log(msg, data=None, hyp=None):
    try:
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":hyp or "A","location":"comment_and_print_functions.py:11","message":msg,"data":data or {},"timestamp":int(time.time()*1000)})+"\n")
    except: pass
# #endregion

def extract_functions_from_cell(cell_source: List[str]) -> List[Dict]:
    """Extract all functions from a cell's source."""
    if not isinstance(cell_source, list):
        cell_source = cell_source.split('\n') if isinstance(cell_source, str) else []
    
    source_text = ''.join(cell_source)
    lines = source_text.split('\n')
    functions = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        func_match = re.match(r'(\s*)def\s+(\w+)\s*\([^)]*\)\s*:', line)
        if func_match:
            indent = func_match.group(1)
            func_name = func_match.group(2)
            
            # Extract function body
            func_lines = [line]
            func_indent_level = len(indent)
            i += 1
            
            # Collect function body
            while i < len(lines):
                current_line = lines[i]
                if not current_line.strip():
                    func_lines.append(current_line)
                    i += 1
                    continue
                
                current_indent = len(current_line) - len(current_line.lstrip())
                
                # Stop if we hit another function/class at same or lower indent
                if current_indent <= func_indent_level:
                    if current_line.strip().startswith('def ') or current_line.strip().startswith('class '):
                        break
                
                func_lines.append(current_line)
                i += 1
                
                # Reasonable limit
                if len(func_lines) > 200:
                    break
            
            func_body = '\n'.join(func_lines)
            has_docstring = '"""' in func_body[:500] or "'''" in func_body[:500]
            
            functions.append({
                'name': func_name,
                'signature': line,
                'body': func_body,
                'has_docstring': has_docstring,
                'indent': indent
            })
            continue
        
        i += 1
    
    return functions

def get_function_description(func_name: str) -> str:
    """Get description based on function name."""
    descriptions = {
        'identify_phases_with_frequency': 'Identifies Exploitation and Exploration phases using semantic similarity and/or word frequency.',
        'analyze_responses': 'Analyzes participant responses to identify phases and calculate exploitation/exploration metrics.',
        'analyze_responses_with_frequency': 'Enhanced analysis including word frequency information.',
        'cosine_similarity': 'Calculates cosine similarity between two vectors.',
        'calculate_novelty': 'Calculates novelty scores for items based on similarity to previous items.',
        'calculate_phase_proximities': 'Calculates proximity metrics between phases (intra-phase and inter-phase similarities).',
        'calculate_phase_centroids': 'Calculates centroid vectors for each phase.',
        'calculate_intra_phase_similarities': 'Calculates similarities within phases.',
        'calculate_inter_phase_similarities': 'Calculates similarities between consecutive phases.',
        'plot_similarity_all_participants_composite': 'Creates composite plot of similarity scores for all participants.',
        'plot_phase_transitions_all_participants_composite_improved': 'Creates composite plot of phase transitions for all participants.',
        'compute_ee_correlations': 'Computes correlations between exploitation/exploration metrics.',
        'run_comprehensive_ee_analysis': 'Runs comprehensive exploitation/exploration analysis.',
    }
    
    if func_name in descriptions:
        return descriptions[func_name]
    
    # Pattern matching
    if 'plot' in func_name.lower():
        return f'Creates visualization for {func_name.replace("plot_", "").replace("_", " ")}.'
    if 'calculate' in func_name.lower():
        return f'Calculates {func_name.replace("calculate_", "").replace("_", " ")}.'
    if 'analyze' in func_name.lower():
        return f'Analyzes {func_name.replace("analyze_", "").replace("_", " ")}.'
    
    return f'Performs {func_name.replace("_", " ")} operation.'

def add_comments_to_functions(notebook_path: str):
    """Add comments to all functions in notebook."""
    # #region agent log
    _log("add_comments_to_functions() entry", {"notebook_path": notebook_path, "cwd": os.getcwd()}, "A")
    # #endregion
    # #region agent log
    _log("Opening notebook file", {"notebook_path": notebook_path}, "C")
    # #endregion
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    all_functions = []
    
    for cell_idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            if not isinstance(source, list):
                continue
            
            # Extract functions
            functions = extract_functions_from_cell(source)
            
            if functions:
                # Rebuild cell source with comments
                source_text = ''.join(source)
                lines = source_text.split('\n')
                new_lines = []
                
                i = 0
                while i < len(lines):
                    line = lines[i]
                    func_match = re.match(r'(\s*)def\s+(\w+)\s*\([^)]*\)\s*:', line)
                    
                    if func_match:
                        indent = func_match.group(1)
                        func_name = func_match.group(2)
                        
                        # Find this function in our extracted list
                        func_info = next((f for f in functions if f['name'] == func_name), None)
                        
                        if func_info and not func_info['has_docstring']:
                            # Add header comment
                            new_lines.append('')
                            new_lines.append(f'{indent}# {"=" * 70}')
                            new_lines.append(f'{indent}# FUNCTION: {func_name}')
                            new_lines.append(f'{indent}# {"=" * 70}')
                            description = get_function_description(func_name)
                            new_lines.append(f'{indent}# {description}')
                            new_lines.append('')
                        
                        new_lines.append(line)
                        
                        # Check if next line is docstring
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if not (next_line.startswith('"""') or next_line.startswith("'''")):
                                # Add docstring
                                if func_info and not func_info['has_docstring']:
                                    doc_indent = indent + '    '
                                    new_lines.append(f'{doc_indent}"""')
                                    new_lines.append(f'{doc_indent}{description}')
                                    new_lines.append(f'{doc_indent}"""')
                        
                        i += 1
                        continue
                    
                    new_lines.append(line)
                    i += 1
                
                # Update cell
                notebook['cells'][cell_idx]['source'] = [l + '\n' for l in new_lines]
                
                for func in functions:
                    func['cell'] = cell_idx
                    all_functions.append(func)
    
    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    return all_functions

def print_all_functions(all_functions: List[Dict], output_file: str):
    """Print all functions to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ALL FUNCTIONS IN NOTEBOOK\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total functions: {len(all_functions)}\n")
        f.write(f"Functions with docstrings: {sum(1 for func in all_functions if func['has_docstring'])}\n")
        f.write(f"Functions needing docstrings: {sum(1 for func in all_functions if not func['has_docstring'])}\n\n")
        
        for i, func in enumerate(all_functions, 1):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"FUNCTION {i}/{len(all_functions)}: {func['name']}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Location: Cell {func['cell']}\n")
            f.write(f"Has Docstring: {'Yes' if func['has_docstring'] else 'No'}\n")
            f.write(f"\n{func['body']}\n")
            f.write("\n" + "-" * 80 + "\n")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Add comments/docstrings to all functions in a Jupyter notebook and print them to a summary file."
    )
    parser.add_argument(
        'notebook',
        type=str,
        nargs='?',
        default='MEDIATION_WITH_NEW_SCORES_10_16.ipynb',
        help='Path to the Jupyter notebook file (default: MEDIATION_WITH_NEW_SCORES_10_16.ipynb)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='all_functions_summary.txt',
        help='Output file for function summary (default: all_functions_summary.txt)'
    )
    return parser.parse_args()

if __name__ == '__main__':
    # #region agent log
    _log("__main__ entry", {"cwd": os.getcwd(), "script_dir": os.path.dirname(os.path.abspath(__file__))}, "B")
    # #endregion
    args = parse_args()
    notebook_path = args.notebook
    output_file = args.output
    # #region agent log
    _log("Arguments parsed", {"notebook_path": notebook_path, "output_file": output_file, "file_exists": os.path.exists(notebook_path)}, "B")
    # #endregion
    
    # Check if file exists before processing
    if not os.path.exists(notebook_path):
        # #region agent log
        abs_path = os.path.abspath(notebook_path)
        _log("File not found in main - raising error", {"notebook_path": notebook_path, "absolute_path": abs_path}, "C")
        # #endregion
        print(f"Error: Notebook file not found: {notebook_path}", file=sys.stderr)
        print(f"  Absolute path: {os.path.abspath(notebook_path)}", file=sys.stderr)
        print(f"  Current working directory: {os.getcwd()}", file=sys.stderr)
        # Check for .ipynb files in current directory
        cwd_notebooks = [f for f in os.listdir('.') if f.endswith('.ipynb')]
        if cwd_notebooks:
            print(f"\n  Found .ipynb files in current directory:", file=sys.stderr)
            for nb in cwd_notebooks:
                print(f"    - {nb}", file=sys.stderr)
        sys.exit(1)
    
    # #region agent log
    _log("File exists, proceeding with processing", {"notebook_path": notebook_path}, "C")
    # #endregion
    
    print("=" * 80)
    print("PROCESSING NOTEBOOK FUNCTIONS")
    print("=" * 80)
    print()
    
    # Add comments
    print("Step 1: Adding comments to functions...")
    # #region agent log
    _log("About to call add_comments_to_functions()", {"notebook_path": notebook_path}, "A")
    # #endregion
    all_functions = add_comments_to_functions(notebook_path)
    print(f"✓ Found {len(all_functions)} functions")
    print()
    
    # Print functions
    print("Step 2: Creating function summary file...")
    print_all_functions(all_functions, output_file)
    print(f"✓ Saved to {output_file}")
    print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total functions: {len(all_functions)}")
    print(f"Functions with docstrings: {sum(1 for f in all_functions if f['has_docstring'])}")
    print(f"Functions needing docstrings: {sum(1 for f in all_functions if not f['has_docstring'])}")
    print()
    print("First 10 functions:")
    for i, func in enumerate(all_functions[:10], 1):
        print(f"  {i:2d}. {func['name']:40s} (Cell {func['cell']:3d}, Docstring: {func['has_docstring']})")

