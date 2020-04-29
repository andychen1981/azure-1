class Recipe(db.Model):
 
    __tablename__ = "recipes"
 
    id = db.Column(db.Integer, primary_key=True)
    recipe_title = db.Column(db.String, nullable=False)
    recipe_description = db.Column(db.String, nullable=False)
    is_public = db.Column(db.Boolean, nullable=False)
    image_filename = db.Column(db.String, default=None, nullable=True)
    image_url = db.Column(db.String, default=None, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
 
    def __init__(self, title, description, user_id, is_public, image_filename=None, image_url=None):
        self.recipe_title = title
        self.recipe_description = description
        self.is_public = is_public
        self.image_filename = image_filename
        self.image_url = image_url
        self.user_id = user_id
 
    def __repr__(self):
        return '<id: {}, title: {}, user_id: {}>'.format(self.id, self.recipe_title, self.user_id)
