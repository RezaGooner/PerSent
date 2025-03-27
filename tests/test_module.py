import pytest
import pandas as pd
import os
from persent import CommentAnalyzer
import shutil

@pytest.fixture
def analyzer(tmp_path):
    """Fixture for creating analyzer with temp model directory"""
    model_dir = os.path.join(tmp_path, "model")
    return CommentAnalyzer(model_dir=model_dir)

@pytest.fixture
def sample_data(tmp_path):
    """Create sample CSV data for testing"""
    data = {
        'body': [
            "این محصول عالی است واقعا توصیه می‌کنم",
            "بدترین خرید عمرم بود",
            "نظر خاصی ندارم",
            "مثل همیشه خوب بود"
        ],
        'recommendation_status': [
            "recommended",
            "not_recommended",
            "no_idea",
            "recommended"
        ]
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join(tmp_path, "sample.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

def test_preprocess_text(analyzer):
    """Test text preprocessing"""
    text = "این یک متن تستی با علائم !؟ و اعداد ۱۲۳ است"
    processed = analyzer._preprocess_text(text)
    assert isinstance(processed, list)
    assert all(isinstance(token, str) for token in processed)
    assert "۱۲۳" not in " ".join(processed)  # Numbers should be removed
    assert "!" not in " ".join(processed)   # Punctuation should be removed

def test_train_and_predict(analyzer, sample_data):
    """Test training and prediction"""
    accuracy = analyzer.train(sample_data)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1
    
    # Test predictions
    assert analyzer.predict("خیلی خوب بود") == "recommended"
    assert analyzer.predict("افتضاح بود") == "not_recommended"
    assert analyzer.predict("نمیدونم") == "no_idea"

def test_model_saving_loading(analyzer, sample_data, tmp_path):
    """Test model saving and loading"""
    analyzer.train(sample_data)
    
    # Test prediction before saving
    pred1 = analyzer.predict("عالی")
    
    # Save and reload
    analyzer.save_model()
    analyzer.load_model()
    
    # Test prediction after loading
    pred2 = analyzer.predict("عالی")
    assert pred1 == pred2

def test_csv_predict(analyzer, sample_data, tmp_path):
    """Test batch CSV prediction"""
    analyzer.train(sample_data)
    
    # Create test CSV
    test_data = pd.DataFrame({
        'comments': [
            "خیلی بد",
            "عالی بود",
            "نظری ندارم"
        ]
    })
    test_csv = os.path.join(tmp_path, "test.csv")
    test_data.to_csv(test_csv, index=False)
    
    # Run prediction
    output_path = os.path.join(tmp_path, "output.csv")
    result = analyzer.csvPredict(
        input_csv=test_csv,
        output_path=output_path,
        summary_path=os.path.join(tmp_path, "summary.csv"),
        text_column='comments'
    )
    
    # Verify results
    assert os.path.exists(output_path)
    output_df = pd.read_csv(output_path)
    assert 'sentiment' in output_df.columns
    assert len(output_df) == 3
    
    # Verify summary
    summary_path = os.path.join(tmp_path, "summary.csv")
    assert os.path.exists(summary_path)
    summary_df = pd.read_csv(summary_path)
    assert len(summary_df) == 5  # 3 categories + total + accuracy

def test_error_handling(analyzer):
    """Test error cases"""
    # Untrained model prediction
    with pytest.raises(Exception, match="Model not trained"):
        analyzer.predict("test")
    
    # Invalid CSV column
    with pytest.raises(ValueError):
        analyzer.csvPredict("nonexistent.csv", "output.csv", text_column=99)

def test_edge_cases(analyzer, sample_data):
    """Test edge cases"""
    analyzer.train(sample_data)
    
    # Empty string
    assert analyzer.predict("") == "no_idea"
    
    # Very short text
    assert analyzer.predict("خوب") in ["recommended", "not_recommended", "no_idea"]
    
    # Foreign language text
    assert isinstance(analyzer.predict("This is English text"), str)
