#!/bin/bash
numbers=($(shuf -i1-4345))
for i in {0..4344}
do
  num=${numbers[$i]}
  filename="diagram${num}"
  stringname="diagram_string${num}"
  pdflatex -escape-shell "data/diagram_files/${filename}.tex"
  convert -density 600x600 "${filename}.pdf" -quality 90 -background white -alpha remove -resize 270x200 "${filename}.png"
  if [ "$i" -lt 4000 ]
  then
      mv "${filename}.png" "data/train_images/${filename}.png"
      if [ "$num" -le 4000 ]
      then
	  mv "data/train_strings/${stringname}.tex" "data/train_strings/${stringname}.tex"
      elif [ "$num" -le 4175 ]
      then
	  mv "data/dev_strings/${stringname}.tex" "data/train_strings/${stringname}.tex"
      else
          mv "data/test_strings/${stringname}.tex" "data/train_strings/${stringname}.tex"
      fi
  elif [ "$i" -lt 4175 ]
  then
      mv "${filename}.png" "data/dev_images/${filename}.png"
      if [ "$num" -le 4000 ]
      then
          mv "data/train_strings/${stringname}.tex" "data/dev_strings/${stringname}.tex"
      elif [ "$num" -le 4175 ]
      then
          mv "data/dev_strings/${stringname}.tex" "data/dev_strings/${stringname}.tex"
      else
          mv "data/test_strings/${stringname}.tex" "data/dev_strings/${stringname}.tex"
      fi
  else
      mv "${filename}.png" "data/test_images/${filename}.png"
      if [ "$num" -le 4000 ]
      then
          mv "data/train_strings/${stringname}.tex" "data/test_strings/${stringname}.tex"
      elif [ "$num" -le 4175 ]
      then
          mv "data/dev_strings/${stringname}.tex" "data/test_strings/${stringname}.tex"
      else
          mv "data/test_strings/${stringname}.tex" "data/test_strings/${stringname}.tex"
      fi
  fi
  rm "${filename}.aux"
  rm "${filename}.log"
  rm "${filename}.pdf"
done
