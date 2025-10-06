import os
import pysrt


def convert_srt_to_plaintext(
        root_folder="commentary_data",
        input_ext=".en.srt",
        output_ext=".en.plain.txt"
):
    """
    Converts all *.en.srt files to *.en.plain.txt in the same folder.
    """
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(input_ext):
                input_path = os.path.join(dirpath, filename)
                output_path = os.path.join(
                    dirpath,
                    filename.replace(input_ext, output_ext)
                )

                try:
                    subs = pysrt.open(input_path)
                    text = " ".join(sub.text.strip() for sub in subs)
                except Exception as e:
                    print(f"Error reading {input_path}: {e}")
                    continue

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)

                print(f"{os.path.basename(output_path)} written")


if __name__ == "__main__":
    convert_srt_to_plaintext()
