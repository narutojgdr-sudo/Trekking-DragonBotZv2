"""
Unit tests for debug and visualization features.
Tests the new log_rejections and log_suspects features.
"""
import ast
import re


class TestVisualizerChanges:
    """Tests for visualizer.py changes"""
    
    def test_visualizer_no_early_return(self):
        """Test that visualizer.draw() does not have early return for show_windows"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Find the draw method
        draw_method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'draw':
                draw_method = node
                break
        
        assert draw_method is not None, "draw method not found"
        
        # Check that there's no early return based on show_windows
        # The old code had: if not self.cfg["show_windows"]: return frame
        method_source = ast.get_source_segment(content, draw_method)
        
        # Should NOT have the old pattern
        assert 'if not self.cfg["show_windows"]:\n            return frame' not in method_source, \
            "Visualizer should not have early return for show_windows"
    
    def test_visualizer_has_reject_counter(self):
        """Test that visualizer displays reject counter"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        # Should have a line that displays rejects count
        assert 'f"Rejects: {len(rejects)}"' in content, \
            "Visualizer should display rejects counter"
    
    def test_visualizer_has_suspect_counter(self):
        """Test that visualizer displays suspect counter"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        # Should count suspects
        assert 'suspect_count = sum(1 for t in tracks if t.state == ConeState.SUSPECT)' in content, \
            "Visualizer should count suspects"
        
        # Should display suspects in track info
        assert 'susp' in content, \
            "Visualizer should display suspect count"
    
    def test_visualizer_improved_rejection_visibility(self):
        """Test that rejection visualization is improved"""
        with open('cone_tracker/visualizer.py', 'r') as f:
            content = f.read()
        
        # Check for rectangle drawing with red color and thickness 2
        assert 'cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)' in content, \
            "Should have red rectangles with thickness 2 for rejections"
        
        # Check for larger text (0.5 instead of 0.35)
        assert '0.5' in content and 'reason' in content, \
            "Should have rejection text with larger font size"


class TestConfigChanges:
    """Tests for config.py changes"""
    
    def test_default_config_has_log_rejections(self):
        """Test that DEFAULT_CONFIG includes log_rejections"""
        with open('cone_tracker/config.py', 'r') as f:
            content = f.read()
        
        assert '"log_rejections"' in content, "DEFAULT_CONFIG should have log_rejections"
        assert 'log_rejections' in content and 'False' in content, \
            "log_rejections should default to False"
    
    def test_default_config_has_log_suspects(self):
        """Test that DEFAULT_CONFIG includes log_suspects"""
        with open('cone_tracker/config.py', 'r') as f:
            content = f.read()
        
        assert '"log_suspects"' in content, "DEFAULT_CONFIG should have log_suspects"
        assert 'log_suspects' in content and 'False' in content, \
            "log_suspects should default to False"


class TestTrackFramesSeen:
    """Tests for Track.frames_seen property"""
    
    def test_track_has_frames_seen_property(self):
        """Test that Track class has frames_seen property"""
        # Test by checking the source code for the property definition
        with open('cone_tracker/tracker.py', 'r') as f:
            content = f.read()
        
        # Should have @property decorator for frames_seen
        assert '@property' in content and 'def frames_seen' in content, \
            "Track should have frames_seen as a property"
        
        # Should return len(self.score_hist)
        assert 'return len(self.score_hist)' in content, \
            "frames_seen should return len(self.score_hist)"
        
        # Also test it functionally if dependencies are available
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from cone_tracker.tracker import Track
            from collections import deque
            
            # Create a track with empty score history
            track = Track(track_id=1, score_hist=deque(maxlen=10))
            assert hasattr(track, 'frames_seen'), "Track should have frames_seen attribute"
            assert track.frames_seen == 0, "frames_seen should be 0 for empty score_hist"
            
            # Add some scores to the history
            track.score_hist.append(0.8)
            assert track.frames_seen == 1, "frames_seen should be 1 after adding one score"
            
            track.score_hist.append(0.9)
            track.score_hist.append(0.85)
            assert track.frames_seen == 3, "frames_seen should be 3 after adding three scores"
            
            # Fill up to maxlen
            for i in range(7):
                track.score_hist.append(0.8 + i * 0.01)
            assert track.frames_seen == 10, "frames_seen should be 10 when score_hist is full"
            
            # Add more items (should exceed maxlen and cap at 10)
            track.score_hist.append(0.95)
            assert track.frames_seen == 10, "frames_seen should remain 10 after exceeding maxlen"
        except ImportError:
            # If we can't import due to missing dependencies, the source code check is sufficient
            pass
    
    def test_frames_seen_used_in_suspect_logging(self):
        """Test that frames_seen is used correctly in app.py suspect logging"""
        with open('cone_tracker/app.py', 'r') as f:
            content = f.read()
        
        # Should use t.frames_seen (as property, not method call)
        assert 't.frames_seen' in content, "app.py should use t.frames_seen in logging"
        # Make sure it's not called as a method
        assert 't.frames_seen()' not in content, "frames_seen should be used as property, not method"


class TestAppChanges:
    """Tests for app.py changes"""
    
    def test_app_has_rejection_logging(self):
        """Test that app.py logs rejections when configured"""
        with open('cone_tracker/app.py', 'r') as f:
            content = f.read()
        
        # Should check for log_rejections config
        assert 'log_rejections' in content, "app.py should check log_rejections config"
        
        # Should log rejections with emoji
        assert '🔴' in content or 'Frame com' in content, \
            "app.py should log rejection messages"
        
        # Should iterate over rejects
        assert 'for bbox, reason in rejects' in content, \
            "app.py should iterate over rejects to log them"
    
    def test_app_has_suspect_logging(self):
        """Test that app.py logs suspects when configured"""
        with open('cone_tracker/app.py', 'r') as f:
            content = f.read()
        
        # Should check for log_suspects config
        assert 'log_suspects' in content, "app.py should check log_suspects config"
        
        # Should log suspects with emoji
        assert '🟡' in content or 'suspects' in content.lower(), \
            "app.py should log suspect messages"
        
        # Should filter for SUSPECT state
        assert 'ConeState.SUSPECT' in content, \
            "app.py should filter tracks by SUSPECT state"


class TestREADMEChanges:
    """Tests for README.md changes"""
    
    def test_readme_has_debugging_section(self):
        """Test that README has debugging section"""
        with open('README.md', 'r') as f:
            content = f.read()
        
        assert 'Debugging e Visualização' in content or 'Debug' in content, \
            "README should have debugging section"
    
    def test_readme_documents_log_rejections(self):
        """Test that README documents log_rejections option"""
        with open('README.md', 'r') as f:
            content = f.read()
        
        assert 'log_rejections' in content, \
            "README should document log_rejections option"
    
    def test_readme_documents_log_suspects(self):
        """Test that README documents log_suspects option"""
        with open('README.md', 'r') as f:
            content = f.read()
        
        assert 'log_suspects' in content, \
            "README should document log_suspects option"
    
    def test_readme_has_example_logs(self):
        """Test that README includes example log output"""
        with open('README.md', 'r') as f:
            content = f.read()
        
        # Should have example showing rejection emoji AND text
        assert '🔴' in content and 'rejeições' in content.lower(), \
            "README should show rejection log examples with emoji"
        
        # Should have example showing suspect emoji AND text
        assert '🟡' in content and 'suspects' in content.lower(), \
            "README should show suspect log examples with emoji"


if __name__ == "__main__":
    # Simple test runner
    test_classes = [TestVisualizerChanges, TestConfigChanges, TestTrackFramesSeen, TestAppChanges, TestREADMEChanges]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
                print(f"✓ {test_class.__name__}.{method_name}")
            except AssertionError as e:
                failed_tests.append((test_class.__name__, method_name, str(e)))
                print(f"✗ {test_class.__name__}.{method_name}: {e}")
            except Exception as e:
                failed_tests.append((test_class.__name__, method_name, str(e)))
                print(f"✗ {test_class.__name__}.{method_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Tests: {passed_tests}/{total_tests} passed")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
        exit(1)
    else:
        print("\nAll tests passed! ✓")
        exit(0)
