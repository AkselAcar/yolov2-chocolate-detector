
# Defining YOLO loss class 
class YOLOLoss(nn.Module): 
	def __init__(self): 
		super().__init__() 
		self.mse = nn.MSELoss()                    # for continuous values (x, y, w, h)
		self.bce = nn.BCEWithLogitsLoss()          # applies sigmoid on the prediction and then cross-entropy => for objectness (with logits),
												   # enables to remove false positives because BCE penalizes much more a big prediction for a low ground truth than the other way.
		self.cross_entropy = nn.CrossEntropyLoss() # penalizes if the logit of the correct class is too low. It makes the prediction converge to a one hot encoding vector AFTER BEING PASSED IN A SOFTMAX.
		self.sigmoid = nn.Sigmoid()                # forces data between 0 and 1
		
		self.lambda_box = 1.0 # weight for box coordinates loss
		self.lambda_object = 2.0 # weight for object loss
		self.lambda_no_object = 3.0 # weight for no object loss
		self.lambda_class = 1.0 # weight for class loss
	
	def forward(self, pred, target, scaled_anchors=SCALED_ANCHORS):
		"""
		Inputs : 
			pred : output of the model, tensor of shape (batch_size, num_anchors, grid_height, grid_width, 5 + num_classes)), raw logits, unnormalized
			target : target from __getitem__ of the dataloader, tensor of shape (batch_size, num_anchors, grid_height, grid_width, 6) 
					 with 6 for (objectness, x, y, w, h, class_label), (w,y,w,h) between 0 and 1, (x,y) relatives to the cell, (w,h) relatives to the grid size, class label between 0 and num_classes-1
			scaled_anchors : tensor of shape (num_anchors, 2), with the anchors normalized relatives to the GRID SIZE (so x between 0 and 18, y between 0 and 12)
		Output :
			total loss of YOLO network (scalar)
		"""
		
		# Identifying which cells in target have objects (and what is the best anchor, the one to detect)
		# and which have no objects
		obj = target[..., 0] == 1
		no_obj = target[..., 0] == 0

		# Reshaping anchors to match predictions. pred.size(1) = num_anchors ==> scaled_anchors.shape becomes (., 9, ., ., 2) which is compatible with pred[..., 3:5].shape to multiply them below
		scaled_anchors = scaled_anchors.reshape(1, pred.size(1), 1, 1, 2).to(pred.device) 

		###################################################### Calculating the BOX LOSS #####################################################        
		# Predicted box coordinates 
		pred_copy_for_box_loss = pred.clone()
		pred_copy_for_box_loss[..., 1:3] = self.sigmoid(pred_copy_for_box_loss[..., 1:3])
		# Target box coordinates
		target_copy_for_box_loss = target.clone()
		target_copy_for_box_loss[..., 3:5] = torch.log(1e-6 + target_copy_for_box_loss[..., 3:5] / scaled_anchors)
		# Calculating box coordinate loss 
		box_loss = self.mse(pred_copy_for_box_loss[..., 1:5][obj], target_copy_for_box_loss[..., 1:5][obj])
		# Goal : (x,y) post-sigmoid are gonna be close to (x,y) of the target,
		# and (w,h) post-processed (so after exponential and multiplication with the scaled_anchor) are gonna be close to the targeted scaled_anchor (the closest to the real target, that has been defined in the dataset __getitem__)

		################################################### Calculating the NO OBJECT LOSS ###################################################
		# Applies, for the cells with no object, the binary cross entropy (bce) between the predicted score (raw logit) and the label (0)
		# Makes the model learn to put a small value in the score where there is no object (so sigmoid(score) is close to zero)
		no_object_loss = self.bce(pred[..., 0:1][no_obj], target[..., 0:1][no_obj])

		##################################################### Calculating the OBJECT LOSS ####################################################
		
		# Clone not to modify the original with sigmoid and exponential because that could cause problems
		pred_copy_for_object_loss = pred.clone()
		pred_xy_01_for_object_loss = self.sigmoid(pred_copy_for_object_loss[..., 1:3])
		pred_wh_ratio_for_object_loss = torch.exp(pred_copy_for_object_loss[..., 3:5]) * scaled_anchors
		# the model predicts w (and h) such that when we do exp(w) (or h) * the predicted scaled_anchors (thanks to the mask "obj"), we get the width (and height) of the box to plot, relative to the grid size (hopefully close to the (w,h) of the targeted scaled_anchor)
		box_preds_for_object_loss = torch.cat([pred_xy_01_for_object_loss, pred_wh_ratio_for_object_loss],dim=-1)

		# Calculating intersection over union of the prediction processed and the target 
		ious = iou(box_preds_for_object_loss[obj], target[..., 1:5][obj]).detach()
		pred_score_for_object_loss = self.sigmoid(pred_copy_for_object_loss[..., 0:1]) # pred[..., 0:1] = score (logit) of the object
		
		object_loss = self.mse(pred_score_for_object_loss[obj], ious * target[...,0:1][obj])
		# Goal : where there is an object, the predicted score after sigmoid has to be close to the IoU of the predicted box (post processed) and the target (the best anchor). At the end it should be one.

		##################################################### Calculating the CLASS LOSS ####################################################  
		# Claculating class loss 
		class_loss = self.cross_entropy((pred[..., 5:][obj]), 
										 target[..., 5][obj].long()) 

	
		# Total loss 
		return ( 
			self.lambda_box * box_loss 
			+ self.lambda_object * object_loss 
			+ self.lambda_no_object * no_object_loss 
			+ self.lambda_class * class_loss 
		)