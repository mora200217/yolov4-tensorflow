import tensorflow as tf 

def process_txt(content): 
    """
        Take the TXT file and read. Extract numpy value,
    """
    txt = tf.io.read_file(content, name = "loaded_txt_file")
    
    txt = tf.strings.split(txt, sep = "\n")
    txt = tf.strings.split(txt, sep = " ")

    a = tf.reduce_all(tf.equal(txt[-1],''))
    def f1(): return txt[:-1]
    def f2(): return txt
    result = tf.cond(a,f1,f2)
    txt = tf.strings.to_number(result)

    print(txt)
    return txt


def process_img(content, new_size = [128, 128]): 
    """
        Read the image file, decode and process
    """
    img = tf.io.read_file(content)
    img = tf.io.decode_png(img)

    img = tf.image.convert_image_dtype(img, tf.float32)
    # img = tf.image.resize(img, new_size)
    img = tf.scalar_mul(tf.constant(1/255), img, name = "img")
    # Text read

    return img, new_size

def process_zip(_img, _txt): 
    """
        Process de Zip dataset 
    """
    img, new_size = process_img(_img)
    txt = process_txt(_txt)
    
    return (img, txt, new_size)