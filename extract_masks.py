# pip install segments-ai
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
import sys

# Initialize a SegmentsDataset from the release file
releases = ['Field_road-v0.field.json', 'Field_road-v0.road.json', 'Field_road-v0.test.json']
for release in releases:
	dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

	# Export to semantic-color format
	export_dataset(dataset, export_format='semantic-color')
