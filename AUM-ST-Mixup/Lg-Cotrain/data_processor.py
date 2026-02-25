import torch
from torch.utils.data import Dataset
import re

import os
from PIL import Image
from torchvision import transforms


class BaseDatasetProcessor:
    def process_dataframe(self, dataframe):
        raise NotImplementedError("Subclasses should implement this method")

    def extract_int_from_string(self, s):
        if isinstance(s, int):
            return s
        if isinstance(s, float):
            # You can choose how to handle floats: either convert if whole number, or ignore
            return int(s)
        if isinstance(s, str):
            match = re.search(r'\d+', s)
            return int(match.group()) if match else None
        return None


class GenericLabelProcessor:
    def get_numeric_label(self, label, label_map):
        if isinstance(label, int) and label in label_map.values():
            return label
        elif isinstance(label, str) and label.isdigit():
            return int(label)
        return label_map.get(label, -1)

    def get_textual_label(self, label, idx_to_label):
        return idx_to_label.get(label, -1)


class TextOnlyProcessor(BaseDatasetProcessor, GenericLabelProcessor):
    def __init__(self, label_map=None, dataset='humanitarian'):
        self.label_map = label_map
        # if self.dataset == 'informative':
        #self.label_map = { "informative": 1, "not_informative": 0 } # if label_map is None else label_map
        # if self.dataset ==  'humanitarian8':
        #     self.label_map = {
        #             "caution_and_advice":0,
        #             "displaced_people_and_evacuations":1,
        #             "infrastructure_and_utility_damage":2,
        #             "not_humanitarian":3,
        #             "other_relevant_information":4,
        #             "requests_or_urgent_needs":5,
        #             "rescue_volunteering_or_donation_effort":6,
        #             "sympathy_and_support":7,
        #         }
        # elif self.dataset == 'humanitarian9':
        #     self.label_map = {
        #             "caution_and_advice":0,
        #             "displaced_people_and_evacuations":1,
        #             "infrastructure_and_utility_damage":2,
        #             "injured_or_dead_people":3,
        #             "not_humanitarian":4,
        #             "other_relevant_information":5,
        #             "requests_or_urgent_needs":6,
        #             "rescue_volunteering_or_donation_effort":7,
        #             "sympathy_and_support":8,
        #         }
        # elif self.dataset == 'humanitarian10':
        #     self.label_map = {
        #             "caution_and_advice":0,
        #             "displaced_people_and_evacuations":1,
        #             "infrastructure_and_utility_damage":2,
        #             "injured_or_dead_people":3,
        #             "missing_or_found_people":4,
        #             "not_humanitarian":5,
        #             "other_relevant_information":6,
        #             "requests_or_urgent_needs":7,
        #             "rescue_volunteering_or_donation_effort":8,
        #             "sympathy_and_support":9,
        #         }

    def process_dataframe(self, dataframe):
        
        if 'ori' in dataframe.columns:
            dataframe.rename(columns={'ori': 'sentence'}, inplace=True)
        
        if 'idx' in dataframe.columns:
            dataframe.rename(columns={'idx': 'id'}, inplace=True)
        
        dataframe['id'] = dataframe.get('tweet_id') if 'id' in dataframe.columns else dataframe.index
        dataframe['id'] = dataframe['tweet_id'].apply(self.extract_int_from_string)
        dataframe['tweet_text'] = dataframe['tweet_text']
        dataframe['label'] = dataframe['label']

        # print(f"Length of Dataframe ------- {len(dataframe)}")
        
        return_keys = ['id', 'tweet_text', 'label']

        # if 'ori_label' in dataframe.columns:
        #     dataframe['label'] = dataframe['ori_label'] #.apply(lambda l: self.get_numeric_label(l, self.label_map))
        #     return_keys.append('label')
            
        # if 'label' in dataframe.columns:
        #     dataframe['label'] = dataframe['label'].apply(lambda l: self.get_numeric_label(l, self.label_map))
        #     return_keys.append('label')

        #print(f"Return keys: {dataframe[return_keys]}")
        return dataframe[return_keys]


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, dataset='sci_nli', include_augmented=False):
        self.dataset = dataset
        self.encoder = TextEncoder(tokenizer, max_len)
        self.dataframe = self.get_dataset_processor(dataset).process_dataframe(dataframe)
        self.include_augmented = include_augmented
        # print(f'df columns: {self.dataframe.columns}')
        if self.include_augmented:
            if self.dataset not in ['informative', 'humanitarian', 'humanitarian8', 'humanitarian9', 'humanitarian10']:
                raise ValueError(f"Augmented data is only available for ag_news, yahoo_answers, amazon_review, yelp_review, and aclImdb datasets")
            if 'aug_1' not in self.dataframe.columns:
                raise ValueError(f"Augmented data requested but 'aug_1' column not found in {dataset} dataframe")
            

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        #print(f"-------------------\n row = {row}")
        #print(f"-------------------\n {row.keys()}")
        item = {
            'labels': torch.tensor(row['label'], dtype=torch.long),
            'id': row['id']
        }
        if 'class_label' in row:
            item['labels'] = torch.tensor(row['label'], dtype=torch.long)

        if self.dataset in ['informative', 'humanitarian', 'humanitarian8', 'humanitarian9', 'humanitarian10']:
#            input_ids, token_type_ids, attention_mask = self.encoder.encode_sentence(
            input_ids, attention_mask = self.encoder.encode_sentence(
                str(row['tweet_text'])
            )
            
            
            if self.include_augmented:
                # Weak augmentation (aug_0)
                aug0_input_ids, aug0_token_type_ids, aug0_attention_mask = self.encoder.encode_sentence(
                    str(row['aug_0'])
                )
                # Strong augmentation (aug_1)
                aug1_input_ids, aug1_token_type_ids, aug1_attention_mask = self.encoder.encode_sentence(
                    str(row['aug_1'])
                )
                item.update({
                    'aug0_input_ids': aug0_input_ids,
                    'aug0_token_type_ids': aug0_token_type_ids,
                    'aug0_attention_mask': aug0_attention_mask,
                    'aug1_input_ids': aug1_input_ids,
                    'aug1_token_type_ids': aug1_token_type_ids,
                    'aug1_attention_mask': aug1_attention_mask,
                })
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        # Add the common encoding fields
        item.update({
            'input_ids': input_ids,
            # 'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        })

        return item
    
    def get_dataset_processor(self, dataset):
        label_maps = {
            'informative': {'informative': 1, 'not_informative': 0},
            'humanitarian': {
                "affected_individuals": 0, 
                "rescue_volunteering_or_donation_effort": 1,
                "infrastructure_and_utility_damage": 2,
                "other_relevant_information": 3,
                "not_humanitarian": 4,
            },
            "humanitarian8": {
                "caution_and_advice":0,
                "displaced_people_and_evacuations":1,
                "infrastructure_and_utility_damage":2,
                "not_humanitarian":3,
                "other_relevant_information":4,
                "requests_or_urgent_needs":5,
                "rescue_volunteering_or_donation_effort":6,
                "sympathy_and_support":7,
            },
            "humanitarian9": {
                "caution_and_advice":0,
                "displaced_people_and_evacuations":1,
                "infrastructure_and_utility_damage":2,
                "injured_or_dead_people":3,
                "not_humanitarian":4,
                "other_relevant_information":5,
                "requests_or_urgent_needs":6,
                "rescue_volunteering_or_donation_effort":7,
                "sympathy_and_support":8,
            },
            "humanitarian10": {
                "caution_and_advice":0,
                "displaced_people_and_evacuations":1,
                "infrastructure_and_utility_damage":2,
                "injured_or_dead_people":3,
                "missing_or_found_people":4,
                "not_humanitarian":5,
                "other_relevant_information":6,
                "requests_or_urgent_needs":7,
                "rescue_volunteering_or_donation_effort":8,
                "sympathy_and_support":9,
            },
        }
        processors = {
            'informative': TextOnlyProcessor(label_maps['informative']),
            'humanitarian': TextOnlyProcessor(label_maps['humanitarian']),
            'humanitarian8': TextOnlyProcessor(label_maps['humanitarian8']),
            'humanitarian9': TextOnlyProcessor(label_maps['humanitarian9']),
            'humanitarian10': TextOnlyProcessor(label_maps['humanitarian10']),
        }
        if dataset not in processors:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return processors[dataset]  


class TextEncoder:
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def encode_sentence(self, sentence):
        """Encode a single sentence and return tensors."""
        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt'  # Return PyTorch tensors directly
        )
        return (
            inputs['input_ids'].squeeze(0),
            #inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0),
            inputs['attention_mask'].squeeze(0)
        )

    def encode_pair_inputs(self, sentence1, sentence2):
        """Encode a pair of sentences and return tensors."""
        inputs = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt'  # Return PyTorch tensors directly
        )
        return (
            inputs['input_ids'].squeeze(0),
            #inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0),
            inputs['attention_mask'].squeeze(0)
        )

    def encode_mc_inputs(self, context, start_ending, endings):
        """Encode multiple choice inputs with context and multiple endings."""
#        all_input_ids, all_token_type_ids, all_attention_masks = [], [], []
        all_input_ids, all_attention_masks = [], [], []
        
        for ending in endings:
            full_ending = f"{start_ending} {ending}" if start_ending else ending
            inputs = self.tokenizer.encode_plus(
                context,
                full_ending,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=False,
                return_tensors='pt'  # Return PyTorch tensors directly
            )
            
            all_input_ids.append(inputs['input_ids'].squeeze(0))
            #all_token_type_ids.append(inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0))
            all_attention_masks.append(inputs['attention_mask'].squeeze(0))
#        return torch.stack(all_input_ids), torch.stack(all_token_type_ids), torch.stack(all_attention_masks)            
        return torch.stack(all_input_ids), torch.stack(all_attention_masks)


    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def encode_sentence(self, sentence):
        """Encode a single sentence and return tensors."""
        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt'  # Return PyTorch tensors directly
        )
        return (
            inputs['input_ids'].squeeze(0),
            #inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0),
            inputs['attention_mask'].squeeze(0)
        )

    def encode_pair_inputs(self, sentence1, sentence2):
        """Encode a pair of sentences and return tensors."""
        inputs = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt'  # Return PyTorch tensors directly
        )
        return (
            inputs['input_ids'].squeeze(0),
            #inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0),
            inputs['attention_mask'].squeeze(0)
        )

    def encode_mc_inputs(self, context, start_ending, endings):
        """Encode multiple choice inputs with context and multiple endings."""
#        all_input_ids, all_token_type_ids, all_attention_masks = [], [], []
        all_input_ids, all_attention_masks = [], [], []
        
        for ending in endings:
            full_ending = f"{start_ending} {ending}" if start_ending else ending
            inputs = self.tokenizer.encode_plus(
                context,
                full_ending,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=False,
                return_tensors='pt'  # Return PyTorch tensors directly
            )
            
            all_input_ids.append(inputs['input_ids'].squeeze(0))
            #all_token_type_ids.append(inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0))
            all_attention_masks.append(inputs['attention_mask'].squeeze(0))
#        return torch.stack(all_input_ids), torch.stack(all_token_type_ids), torch.stack(all_attention_masks)            
        return torch.stack(all_input_ids), torch.stack(all_attention_masks)
    




class ImageOnlyProcessor:
    def __init__(self, label_map=None, dataset='informative'):
        self.dataset = dataset
        if dataset == 'informative':
            self.label_map = {
                "informative": 1, 
                "not_informative": 0
            }
        elif dataset == 'humanitarian':
            self.label_map = {
                "affected_individuals": 0, 
                "rescue_volunteering_or_donation_effort": 1,
                "infrastructure_and_utility_damage": 2,
                "other_relevant_information": 3,
                "not_humanitarian": 4,
            }
        elif dataset == 'humanitarian10':
            self.label_map = {
                "caution_and_advice":0,
                "displaced_people_and_evacuations":1,
                "infrastructure_and_utility_damage":2,
                "injured_or_dead_people":3,
                "missing_or_found_people":4,
                "not_humanitarian":5,
                "other_relevant_information":6,
                "requests_or_urgent_needs":7,
                "rescue_volunteering_or_donation_effort":8,
                "sympathy_and_support":9,
            }
        elif dataset == 'humanitarian9':
            self.label_map = {
                "caution_and_advice":0,
                "displaced_people_and_evacuations":1,
                "infrastructure_and_utility_damage":2,
                "injured_or_dead_people":3,
                #"missing_or_found_people":4,
                "not_humanitarian":4,
                "other_relevant_information":5,
                "requests_or_urgent_needs":6,
                "rescue_volunteering_or_donation_effort":7,
                "sympathy_and_support":8,
            }
        elif dataset == 'humanitarian8':
            self.label_map = {
                "caution_and_advice":0,
                "displaced_people_and_evacuations":1,
                "infrastructure_and_utility_damage":2,
                "not_humanitarian":3,
                "other_relevant_information":4,
                "requests_or_urgent_needs":5,
                "rescue_volunteering_or_donation_effort":6,
                "sympathy_and_support":7,
            }
        else:
            self.label_map = label_map if label_map is not None else {}

    def extract_int_from_string(self, s):
        """Extract integer from string if needed."""
        if isinstance(s, int):
            return s
        return int(''.join(filter(str.isdigit, str(s)))) if any(c.isdigit() for c in str(s)) else hash(s)

    def get_numeric_label(self, label, label_map):
        """Convert string label to numeric."""
        if isinstance(label, (int, float)):
            return int(label)
        return label_map.get(label, label)

    def process_dataframe(self, dataframe):
        """Process dataframe to standardize columns."""
        # Standardize ID column
        # if 'idx' in dataframe.columns:
        #     dataframe.rename(columns={'idx': 'id'}, inplace=True)
        
        #dataframe['id'] = dataframe['id']#.apply(self.extract_int_from_string)
        dataframe['id'] = dataframe.index

        # if 'tweet_id' in dataframe.columns:
        #     dataframe['id'] = dataframe['tweet_id'].apply(self.extract_int_from_string)
        # elif 'id' not in dataframe.columns:
        #     dataframe['id'] = dataframe.index
        
        # Standardize image path column
        if 'image_path' not in dataframe.columns:
            if 'img_path' in dataframe.columns:
                dataframe.rename(columns={'img_path': 'image_path'}, inplace=True)
            elif 'image' in dataframe.columns:
                dataframe.rename(columns={'image': 'image_path'}, inplace=True)
        
        # Process labels
        if 'label' in dataframe.columns:
            dataframe['label'] = dataframe['label'].apply(
                lambda l: self.get_numeric_label(l, self.label_map)
            )
        
        return_keys = ['id', 'image_path', 'label']
        
        # Include augmented image paths if available
        if 'aug_0' in dataframe.columns:
            return_keys.append('aug_0')
        if 'aug_1' in dataframe.columns:
            return_keys.append('aug_1')
        
        return dataframe[return_keys]


class ImageEncoder:
    def __init__(self, processor):
        """
        Args:
            processor: Image processor (CLIPProcessor, AutoImageProcessor, etc.)
        """
        self.processor = processor

    def encode_image(self, image):
        """
        Encode a single image and return pixel values tensor.
        
        Args:
            image: PIL Image object
            
        Returns:
            pixel_values: Tensor of shape (C, H, W)
        """
        inputs = self.processor(
            images=image,
            return_tensors='pt'
        )
        return inputs['pixel_values'].squeeze(0)  # Remove batch dimension

    def encode_image_batch(self, images):
        """
        Encode multiple images and return stacked tensors.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            pixel_values: Tensor of shape (num_images, C, H, W)
        """
        all_pixel_values = []
        
        for image in images:
            inputs = self.processor(
                images=image,
                return_tensors='pt'
            )
            all_pixel_values.append(inputs['pixel_values'].squeeze(0))
        
        return torch.stack(all_pixel_values)


class ImageDataset(Dataset):
    def __init__(
        self, 
        dataframe, 
        processor, 
        dataset='informative', 
        image_root=None,
        include_augmented=False,
        transform=None
    ):
        """
        Args:
            dataframe: DataFrame with image paths and labels
            processor: Image processor (CLIPProcessor, AutoImageProcessor, etc.)
            image_root: Root directory for images
            dataset: Dataset type ('informative', 'humanitarian', etc.)
            include_augmented: Whether to include augmented images
            transform: Optional transforms to apply BEFORE processor (e.g., data augmentation)
        """
        self.dataset = dataset
        self.image_root = image_root if image_root is not None else ""
        self.encoder = ImageEncoder(processor)
        self.dataframe = self.get_dataset_processor(dataset).process_dataframe(dataframe)
        self.include_augmented = include_augmented
        self.transform = transform
        
        if self.include_augmented:
            if 'aug_0' not in self.dataframe.columns or 'aug_1' not in self.dataframe.columns:
                raise ValueError(
                    f"Augmented data requested but 'aug_0' or 'aug_1' columns not found in dataframe"
                )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Base item with label and id
        item = {
            'labels': torch.tensor(row['label'], dtype=torch.long),
            'id': row['id']
        }
        
        # Load and encode main image
        image_path = os.path.join(self.image_root, str(row['image_path']))
        image = Image.open(image_path).convert('RGBA')
        
        # Apply optional transforms (e.g., data augmentation) BEFORE processor
        if self.transform is not None:
            image = self.transform(image)
        
        # Encode with processor
        pixel_values = self.encoder.encode_image(image)
        
        item['pixel_values'] = pixel_values
        
        # Handle augmented images if requested
        if self.include_augmented:
            # Weak augmentation (aug_0)
            aug0_path = os.path.join(self.image_root, str(row['aug_0']))
            aug0_image = Image.open(aug0_path).convert('RGBA')
            if self.transform is not None:
                aug0_image = self.transform(aug0_image)
            aug0_pixel_values = self.encoder.encode_image(aug0_image)
            
            # Strong augmentation (aug_1)
            aug1_path = os.path.join(self.image_root, str(row['aug_1']))
            aug1_image = Image.open(aug1_path).convert('RGBA')
            if self.transform is not None:
                aug1_image = self.transform(aug1_image)
            aug1_pixel_values = self.encoder.encode_image(aug1_image)
            
            item.update({
                'aug0_pixel_values': aug0_pixel_values,
                'aug1_pixel_values': aug1_pixel_values,
            })
        
        return item
    
    def get_dataset_processor(self, dataset):
        """Get appropriate processor for dataset type."""
        label_maps = {
            'informative': {'informative': 1, 'not_informative': 0},
            'humanitarian': {
                "affected_individuals": 0, 
                "rescue_volunteering_or_donation_effort": 1,
                "infrastructure_and_utility_damage": 2,
                "other_relevant_information": 3,
                "not_humanitarian": 4,
            },
        }
        
        processors = {
            'informative': ImageOnlyProcessor(label_maps.get('informative'), dataset='informative'),
            'humanitarian': ImageOnlyProcessor(label_maps.get('humanitarian'), dataset='humanitarian'),
            'humanitarian8': ImageOnlyProcessor(label_maps.get('humanitarian8'), dataset='humanitarian8'),
            'humanitarian9': ImageOnlyProcessor(label_maps.get('humanitarian9'), dataset='humanitarian9'),
            'humanitarian10': ImageOnlyProcessor(label_maps.get('humanitarian10'), dataset='humanitarian10'),
        }
        
        # Allow custom datasets
        if dataset not in processors:
            processors[dataset] = ImageOnlyProcessor(dataset=dataset)
        
        return processors[dataset]

class Old_ImageOnlyProcessor(BaseDatasetProcessor, GenericLabelProcessor):
    def __init__(self, label_map=None):
        # Default to 5-way humanitarian categories if nothing is passed
        if label_map is None:
            self.label_map = { 
                "informative": 1,
                "not_informative": 0
            }
            # self.label_map = {
            #     "affected_individuals": 0,
            #     "rescue_volunteering_or_donation_effort": 1,
            #     "infrastructure_and_utility_damage": 2,
            #     "other_relevant_information": 3,
            #     "not_humanitarian": 4,
            # }
        else:
            self.label_map = label_map

    def _standardize_image_column(self, dataframe):
        """
        Try to infer which column contains image paths and rename to 'image_path'.
        Adapt or simplify this to match your actual columns if needed.
        """
        candidate_cols = [
            "image_path",
            "img_path",
            "image",
            "img",
            "image_name",
            "image_file",
            "jpg",
            "png",
        ]
        existing = [c for c in candidate_cols if c in dataframe.columns]
        if not existing:
            raise ValueError(
                f"No image path column found in dataframe. "
                f"Expected one of: {candidate_cols}"
            )
        main_col = existing[0]
        if main_col != "image_path":
            dataframe.rename(columns={main_col: "image_path"}, inplace=True)
        return dataframe

    def process_dataframe(self, dataframe):
        # Standardize id column
        if "idx" in dataframe.columns:
            dataframe.rename(columns={"idx": "id"}, inplace=True)

        if "id" in dataframe.columns:
            dataframe["id"] = dataframe["id"].apply(self.extract_int_from_string)
        elif "tweet_id" in dataframe.columns:
            dataframe["id"] = dataframe["tweet_id"].apply(self.extract_int_from_string)
        else:
            # Fallback to dataframe index
            dataframe["id"] = dataframe.index

        # Standardize image path column
        dataframe = self._standardize_image_column(dataframe)

        # Standardize label column
        if "label" not in dataframe.columns:
            raise ValueError("Expected a 'label' column in the dataframe for images")

        dataframe["label"] = dataframe["label"].apply(
            lambda l: self.get_numeric_label(l, self.label_map)
        )

        return_keys = ["id", "image_path", "label"]

        # Optionally keep tweet text for debugging or multi modal experiments
        if "tweet_text" in dataframe.columns:
            return_keys.append("tweet_text")

        return dataframe[return_keys]


class Old_ImageEncoder:
    def __init__(self, processor):
        """
        Args:
            processor: CLIPProcessor for image preprocessing
        """
        self.processor = processor
    
    def encode_image(self, image):
        """
        Encode a single image and return tensor.
        
        Args:
            image: PIL Image object
            
        Returns:
            pixel_values: Tensor of shape (3, 224, 224)
        """
        inputs = self.processor(
            images=image,
            return_tensors='pt'
        )
        return inputs['pixel_values'].squeeze(0)  # Remove batch dimension
    
    def encode_image_batch(self, images):
        """
        Encode multiple images and return stacked tensors.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            pixel_values: Tensor of shape (num_images, 3, 224, 224)
        """
        all_pixel_values = []
        
        for image in images:
            inputs = self.processor(
                images=image,
                return_tensors='pt'
            )
            all_pixel_values.append(inputs['pixel_values'].squeeze(0))
        
        return torch.stack(all_pixel_values)


class OldImageDataset(Dataset):
    def __init__(
        self,
        dataframe,
        processor,
        dataset="informative"
    ):
#        image_root=None,
#        transform=None,
#        weak_transform=None,
#        strong_transform=None,
#        include_augmented=False,
        """
        Args:
            dataframe: pandas.DataFrame with image path and label columns.
            image_root: root directory where the images are stored.
            dataset: 'informative' or 'humanitarian'.
            transform: transform for the main image view.
            weak_transform: optional weak augmentation transform.
            strong_transform: optional strong augmentation transform.
            include_augmented: if True, also return weak and strong augmented views.
        """
        self.dataset = dataset
        self.image_root = "/home/b/bharanibala/mllmd/CrisisMMD_v2.0/"
        #self.include_augmented = include_augmented

        self.processor = self.get_dataset_processor(dataset)
        # Do not mutate the original dataframe
        self.dataframe = self.processor.process_dataframe(dataframe.copy())

        # Main encoder
#        self.encoder = ImageEncoder(transform=transform)

        # Augmentation pipelines
#        self.weak_transform = weak_transform
#        self.strong_transform = strong_transform

        # if self.include_augmented:
        #     if self.weak_transform is None:
        #         raise ValueError("include_augmented=True requires weak_transform")
        #     if self.strong_transform is None:
        #         raise ValueError("include_augmented=True requires strong_transform")

    def __len__(self):
        return len(self.dataframe)

    def _full_image_path(self, rel_path):
        # Handle both absolute and relative paths
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(self.image_root, rel_path)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        #img_rel_path = str(row["image_path"])
        image_path = str(row["image_path"]) #self._full_image_path(img_rel_path)

        # Main view
        image_main = self.encoder.encode_image(image_path)

        item = {
            "image": image_main,
            "labels": torch.tensor(row["label"], dtype=torch.long),
            "id": row["id"],
        }

        # Optional augmented views for SSL or consistency regularization
        # if self.include_augmented:
        #     image_raw = Image.open(image_path).convert("RGB")
        #     image_weak = self.weak_transform(image_raw)
        #     image_strong = self.strong_transform(image_raw)
        #     item.update(
        #         {
        #             "image_weak": image_weak,
        #             "image_strong": image_strong,
        #         }
        #     )

        # Pass through tweet text if present
        if "tweet_text" in row:
            item["tweet_text"] = row["tweet_text"]

        return item

    def get_dataset_processor(self, dataset):
        label_maps = {
            "informative": {"informative": 1, "not_informative": 0},
            "humanitarian": {
                "affected_individuals": 0,
                "rescue_volunteering_or_donation_effort": 1,
                "infrastructure_and_utility_damage": 2,
                "other_relevant_information": 3,
                "not_humanitarian": 4,
            },
        }
        if dataset not in label_maps:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return ImageOnlyProcessor(label_map=label_maps[dataset])