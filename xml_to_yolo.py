def convert_xml_to_yolo(xml_folder, image_folder, output_folder, class_dict):
    import os, glob
    from lxml import etree

    os.makedirs(output_folder, exist_ok=True)

    for xml_file in glob.glob(xml_folder + "/*.xml"):
        tree = etree.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        txt_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(output_folder, txt_filename), 'w') as f:
            for obj in root.findall('object'):
                label = obj.find('name').text  # No lower(), no strip()
                print(f"🟡 Found label: '{label}'")

                if label not in class_dict:
                    print(f"❌ Skipping unknown label: '{label}'")
                    continue

                label_id = class_dict[label]
                bbox = obj.find('bndbox')
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))

                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                f.write(f"{label_id} {x_center} {y_center} {box_width} {box_height}\n")
                print(f"✅ Wrote bounding box for '{label}' to {txt_filename}")