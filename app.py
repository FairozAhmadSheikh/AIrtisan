from flask import Flask, render_template, request, redirect, url_for
import os

from style_transfer import run_style_transfer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stylize', methods=['POST'])
def stylize():
    style_img = request.files['style']
    content_img = request.files['content']

    style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_img.filename)
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_img.filename)

    style_img.save(style_path)
    content_img.save(content_path)

    # Placeholder: style transfer code will come later
    output_path = content_path  # For now, just return the content image

    return render_template('result.html', output_image=output_path)


@app.route('/stylize', methods=['POST'])
def stylize():
    style_img = request.files['style']
    content_img = request.files['content']

    style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_img.filename)
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_img.filename)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'stylized_{content_img.filename}')

    style_img.save(style_path)
    content_img.save(content_path)

    run_style_transfer(style_path, content_path, output_path)

    return render_template('result.html', output_image=output_path)

if __name__ == '__main__':
    app.run(debug=True)
