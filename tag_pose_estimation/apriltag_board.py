# Wrapper for cv2.aruco.Board with dictionary as string
import cv2

class AprilTagBoard(cv2.aruco.Board):
	"""
	Wrapper around cv2.aruco.Board that stores the dictionary as a string.
    Blocks generateImage() method since it requires the actual dictionary 
	object.
	"""
	def __init__(self, objPoints, ids, dictionary, **kwargs):
		# Use a default Aruco dictionary for base class
		default_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
		super().__init__(objPoints, default_dict, ids, **kwargs)
		self._dictionary_str = dictionary  # Store the string for metadata
		self._kwargs = kwargs

	def getDictionary(self):
		return self._dictionary_str

	def generateImage(self, *args, **kwargs):
		raise NotImplementedError(
			"generateImage() is blocked: requires actual "
            "Aruco dictionary object."
        )

	def __repr__(self):
		objPoints_str = str(self.getObjPoints())
		ids_str = str(self.getIds())
		return (
			f"AprilTagBoard(objPoints={objPoints_str}, ids={ids_str}, "
	        f"dictionary='{self._dictionary_str}')"
        )
