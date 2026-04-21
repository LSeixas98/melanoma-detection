"""Script para corrigir indentação dos arquivos GUI."""

import re

files = [
    'gui/training_interface.py',
    'gui/prediction_interface.py',
    'gui/comparison_interface.py'
]

for filepath in files:
    print(f"Processando {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Separar por linhas
    lines = content.split('\n')
    fixed_lines = []
    in_function = False
    base_indent = 0

    for i, line in enumerate(lines):
        # Detectar início de função create_*_interface
        if line.strip().startswith('def create_') and '_interface' in line:
            in_function = True
            base_indent = 0
            fixed_lines.append(line)
            continue

        # Detectar fim da função (próxima def ou fim do arquivo)
        if in_function and line.strip().startswith('def ') and 'create_' not in line:
            in_function = False

        if in_function:
            # Ajustar indentação relativa à função
            stripped = line.lstrip()
            if stripped:
                # Calcular indentação correta baseada no conteúdo
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    # Docstring
                    fixed_lines.append('    ' + stripped)
                elif stripped.startswith('gr.'):
                    # Componente Gradio no nível raiz da função
                    fixed_lines.append('    ' + stripped)
                elif stripped.startswith('with gr.'):
                    # Contexto gr no nível raiz
                    fixed_lines.append('    ' + stripped)
                elif stripped.startswith('def '):
                    # Função interna
                    fixed_lines.append('    ' + stripped)
                else:
                    # Manter indentação relativa original
                    indent_level = len(line) - len(stripped)
                    # Ajustar para múltiplos de 4
                    new_indent = ((indent_level // 4) * 4)
                    if new_indent < 4:
                        new_indent = 4
                    fixed_lines.append(' ' * new_indent + stripped)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    # Salvar arquivo corrigido
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))

    print(f"  OK: {filepath} corrigido")

print("\nTodos os arquivos foram corrigidos!")
