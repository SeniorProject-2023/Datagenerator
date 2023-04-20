from pylabel import importer

dataset = importer.ImportCoco(path="/home/astroc/Projects/Python/Graduation/SegmentationGeneration/book.json", 
                              path_to_images="/home/astroc/Projects/Python/Graduation/SegmentationGeneration/book")
dataset.export.ExportToYoloV5(output_path="./yolo/labels", copy_images=True)
