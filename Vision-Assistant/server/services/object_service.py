def detect_objects(frame, object_model):

    results = object_model(frame)

    object_names = set()

    if results[0].boxes is not None:

        for box in results[0].boxes:

            if float(box.conf[0]) > 0.5:

                label = results[0].names[int(box.cls[0])]

                object_names.add(label)

    return list(object_names)