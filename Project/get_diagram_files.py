def get_diagram_files(filename):
    j = 0
    i = 0
    with open(filename) as f:
        lines = f.readlines()
        while i < len(lines):
            if lines[i].strip() == "$$":
                j += 1
                diagram_end = i + 1
                while lines[diagram_end].strip() != "$$":
                    diagram_end += 1
                # diagram = ['\\input{preamble}\n', '\\begin{document}\n']
                diagram = lines[i+1:diagram_end]
                # diagram.append('\\end{document}\n')
                diagram = "".join(diagram)
                if j <= 4000:
                    with open(f'data/train_strings/diagram_string{j}.tex', 'w') as d:
                        d.write(f"{diagram}")
                elif j <= 4175:
                    with open(f'data/dev_strings/diagram_string{j}.tex', 'w') as d:
                        d.write(f"{diagram}")
                else:
                    with open(f'data/test_strings/diagram_string{j}.tex', 'w') as d:
                        d.write(f"{diagram}")
                # with open(f'data/diagram_files/diagram{j}.tex', 'w') as d:
                    # d.write(f"{diagram}")
                if diagram_end == len(lines) - 1:
                    break
                i = diagram_end + 1
            else:
                i += 1

def main():
    get_diagram_files('diagrams.tex')

if __name__ == "__main__":
    main()
