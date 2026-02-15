"""
Scikit-learn Real-World Datasets
fetch_openml, fetch_20newsgroups, fetch_lfw_people
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.datasets import fetch_openml, fetch_20newsgroups, fetch_lfw_people


def main():
    print("=" * 60)
    print("Real-World Datasets: fetch_* functions")
    print("=" * 60)

    print("\n[1] fetch_openml - MNIST (small subset):")
    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        print(f"    X shape: {mnist.data.shape}")
        print(f"    y shape: {mnist.target.shape}")
        print(f"    Data range: [{mnist.data.min():.0f}, {mnist.data.max():.0f}]")
    except Exception as e:
        print(f"    (Skipped - requires download: {e})")

    print("\n[2] fetch_openml - Alternative dataset:")
    try:
        titanic = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
        print(f"    Dataset: titanic")
        print(f"    Frame shape: {titanic.frame.shape if hasattr(titanic, 'frame') else 'N/A'}")
    except Exception as e:
        print(f"    (Skipped: {e})")

    print("\n[3] fetch_20newsgroups - Text data:")
    try:
        news_train = fetch_20newsgroups(subset="train", categories=["sci.med", "sci.space"], shuffle=True, random_state=42)
        print(f"    Subset: train")
        print(f"    Categories: {news_train.target_names}")
        print(f"    Number of documents: {len(news_train.data)}")
        print(f"    First doc (first 80 chars): {news_train.data[0][:80]}...")
    except Exception as e:
        print(f"    (Skipped: {e})")

    print("\n[4] fetch_20newsgroups - Subsets:")
    print("    subset='train' | 'test' | 'all'")
    print("    remove=('headers','footers','quotes') for cleaner text")

    print("\n[5] fetch_lfw_people - Face images:")
    try:
        lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        print(f"    X shape: {lfw.images.shape}")
        print(f"    Images: {lfw.images.shape[1]}x{lfw.images.shape[2]} pixels")
        print(f"    n_samples: {lfw.data.shape[0]}")
        print(f"    n_classes: {len(lfw.target_names)}")
    except Exception as e:
        print(f"    (Skipped - requires download: {e})")

    print("\n[6] Download and cache location:")
    from sklearn.datasets import get_data_home
    print(f"    get_data_home(): {get_data_home()}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
