from os import listdir
import glob

def parse_diagram(lines):
    diagrams = []
    for i in range(len(lines)):
        if lines[i].strip() == "\\xymatrix{":
            diagram_start = i-1
            while lines[diagram_start].strip() != "$$" and lines[diagram_start].strip() != "\\begin{{equation}}":
                diagram_start -= 1
            assert lines[diagram_start].strip() == "$$", "the previous line is " + lines[diagram_start]
            diagram_end = i
            while lines[diagram_end].strip() != "}" and lines[diagram_end].strip() != "$$":
                diagram_end += 1
                print(lines[diagram_end])

            diagram = lines[i:diagram_end+1]
            diagram.insert(0, "$$\n")
            if diagram[-1].strip() == "}":
                diagram.append("$$\n")

            diagram = "".join(diagram)
            diagrams.append(diagram)
            
    return diagrams

def main():
    stacks_project = []
    for file in glob.glob("./stacks-project/*.tex"):
        stacks_project.append(file)
    
    # count = 0
    for filename in stacks_project:
        overwrite = False
        if stacks_project.index(filename) == 0:
            overwrite = True
        with open(filename) as f:
            lines = f.readlines()
            # for line in lines:
            #     if line.strip() == "\\xymatrix{":
            #         count += 1
            diagrams = parse_diagram(lines)
            if overwrite:
                mode = 'w'
            else:
                mode = 'a'
            with open('diagrams.tex', mode) as d:
                for diagram in diagrams:
                    d.write(f"{diagram}\n")
                
    # print("There are %d diagrams" % count)

if __name__ == "__main__":
    main()