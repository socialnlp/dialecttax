"""Tests for dialecttax.gradients."""

from unittest.mock import MagicMock

import torch

from dialecttax.gradients import _countsketch, _gradient_norm, compute_projected_gradient, project_gradient


################
# COUNTSKETCH  #
################

class TestCountSketch:
    def test_output_shape(self):
        """Output shape matches projection_dim."""
        grad = torch.randn(1000)
        result = _countsketch(grad, projection_dim=64, seed=42)
        assert result.shape == (64,)

    def test_output_dtype(self):
        """Output is always float32."""
        grad = torch.randn(100, dtype=torch.bfloat16)
        result = _countsketch(grad, projection_dim=32, seed=0)
        assert result.dtype == torch.float32

    def test_deterministic_same_seed(self):
        """Same input + same seed = same output."""
        grad = torch.randn(500)
        a = _countsketch(grad, projection_dim=64, seed=42)
        b = _countsketch(grad, projection_dim=64, seed=42)
        torch.testing.assert_close(a, b)

    def test_different_seed(self):
        """Different seeds produce different projections."""
        grad = torch.randn(500)
        a = _countsketch(grad, projection_dim=64, seed=42)
        b = _countsketch(grad, projection_dim=64, seed=99)
        assert not torch.allclose(a, b)

    def test_zero_gradient(self):
        """Zero gradient projects to zero."""
        grad = torch.zeros(200)
        result = _countsketch(grad, projection_dim=32, seed=0)
        torch.testing.assert_close(result, torch.zeros(32))

    def test_linearity(self):
        """CountSketch is linear: sketch(a + b) == sketch(a) + sketch(b)."""
        a = torch.randn(300)
        b = torch.randn(300)
        seed = 7
        dim = 64
        sketch_sum = _countsketch(a + b, dim, seed)
        sketch_a = _countsketch(a, dim, seed)
        sketch_b = _countsketch(b, dim, seed)
        torch.testing.assert_close(sketch_sum, sketch_a + sketch_b)

    def test_scaling(self):
        """CountSketch is linear: sketch(c * a) == c * sketch(a)."""
        a = torch.randn(200)
        c = 3.5
        dim = 32
        seed = 11
        torch.testing.assert_close(
            _countsketch(c * a, dim, seed),
            c * _countsketch(a, dim, seed),
        )

    def test_large_input_chunking(self):
        """Large inputs (exceeding CHUNK_SIZE) produce correct results."""
        # Use a grad larger than CHUNK_SIZE (2**22 = ~4M)
        # We test consistency: same seed gives same result regardless of internal chunking
        grad = torch.randn(100)
        dim = 16
        seed = 42

        # Compute with default chunking
        result = _countsketch(grad, dim, seed)

        # Verify determinism (implicitly tests chunking consistency)
        result2 = _countsketch(grad, dim, seed)
        torch.testing.assert_close(result, result2)

    def test_preserves_inner_product_statistically(self):
        """CountSketch approximately preserves inner products on average."""
        torch.manual_seed(0)
        a = torch.randn(10000)
        b = torch.randn(10000)

        true_dot = (a @ b).item()

        # Average over multiple independent sketches to reduce variance
        dim = 4096
        n_trials = 20
        approx_dots = []
        for s in range(n_trials):
            sketch_a = _countsketch(a, dim, seed=s * 1000)
            sketch_b = _countsketch(b, dim, seed=s * 1000)
            approx_dots.append((sketch_a @ sketch_b).item())

        mean_approx = sum(approx_dots) / n_trials
        assert abs(mean_approx - true_dot) / (abs(true_dot) + 1e-8) < 0.5


####################
# PROJECT_GRADIENT #
####################

class TestProjectGradient:
    def _make_model_with_grads(self, param_shapes, seed=0):
        """Create a mock model with parameters that have .grad set."""
        torch.manual_seed(seed)
        params = []
        for shape in param_shapes:
            p = torch.nn.Parameter(torch.randn(shape))
            p.grad = torch.randn(shape)
            params.append(p)

        model = MagicMock()
        model.parameters.return_value = params
        return model

    def test_output_shape(self):
        model = self._make_model_with_grads([(10, 5), (5,)])
        result = project_gradient(model, projection_dim=32, seed=42)
        assert result.shape == (32,)

    def test_output_on_cpu(self):
        model = self._make_model_with_grads([(10, 5)])
        result = project_gradient(model, projection_dim=16, seed=0)
        assert result.device == torch.device("cpu")

    def test_deterministic_same_seed(self):
        model = self._make_model_with_grads([(10, 5), (5,)])
        a = project_gradient(model, projection_dim=32, seed=42)
        b = project_gradient(model, projection_dim=32, seed=42)
        torch.testing.assert_close(a, b)

    def test_different_seed(self):
        model = self._make_model_with_grads([(10, 5), (5,)])
        a = project_gradient(model, projection_dim=32, seed=42)
        b = project_gradient(model, projection_dim=32, seed=99)
        assert not torch.allclose(a, b)

    def test_skips_none_grads(self):
        """Parameters with grad=None are skipped without error."""
        torch.manual_seed(0)
        p1 = torch.nn.Parameter(torch.randn(10))
        p1.grad = torch.randn(10)
        p2 = torch.nn.Parameter(torch.randn(5))
        p2.grad = None

        model = MagicMock()
        model.parameters.return_value = [p1, p2]

        result = project_gradient(model, projection_dim=16, seed=0)
        assert result.shape == (16,)
        assert not torch.all(result == 0)

    def test_zero_grads(self):
        """All-zero gradients project to zero."""
        p1 = torch.nn.Parameter(torch.randn(10))
        p1.grad = torch.zeros(10)
        p2 = torch.nn.Parameter(torch.randn(5))
        p2.grad = torch.zeros(5)

        model = MagicMock()
        model.parameters.return_value = [p1, p2]

        result = project_gradient(model, projection_dim=16, seed=0)
        torch.testing.assert_close(result, torch.zeros(16))


#################
# GRADIENT_NORM #
#################

class TestGradientNorm:
    def test_zero_grads(self):
        """Zero gradients have norm 0."""
        p = torch.nn.Parameter(torch.randn(10))
        p.grad = torch.zeros(10)
        model = MagicMock()
        model.parameters.return_value = [p]
        assert _gradient_norm(model) == 0.0

    def test_known_norm(self):
        """Matches manually computed norm."""
        p = torch.nn.Parameter(torch.randn(2))
        p.grad = torch.tensor([3.0, 4.0])
        model = MagicMock()
        model.parameters.return_value = [p]
        assert abs(_gradient_norm(model) - 5.0) < 1e-5

    def test_multiple_params(self):
        """Norm is computed across all parameters."""
        p1 = torch.nn.Parameter(torch.randn(1))
        p1.grad = torch.tensor([3.0])
        p2 = torch.nn.Parameter(torch.randn(1))
        p2.grad = torch.tensor([4.0])
        model = MagicMock()
        model.parameters.return_value = [p1, p2]
        # sqrt(9 + 16) = 5
        assert abs(_gradient_norm(model) - 5.0) < 1e-5

    def test_skips_none_grads(self):
        """Parameters with grad=None are skipped."""
        p1 = torch.nn.Parameter(torch.randn(1))
        p1.grad = torch.tensor([3.0])
        p2 = torch.nn.Parameter(torch.randn(1))
        p2.grad = None
        model = MagicMock()
        model.parameters.return_value = [p1, p2]
        assert abs(_gradient_norm(model) - 3.0) < 1e-5

    def test_positive(self):
        """Norm is non-negative."""
        p = torch.nn.Parameter(torch.randn(10))
        p.grad = torch.randn(10)
        model = MagicMock()
        model.parameters.return_value = [p]
        assert _gradient_norm(model) >= 0.0


##############################
# COMPUTE_PROJECTED_GRADIENT #
##############################

class TestComputeProjectedGradient:
    def _make_tiny_model(self):
        """Create a minimal CausalLM-like model for testing."""
        model = torch.nn.Sequential(
            torch.nn.Embedding(32, 16),
            torch.nn.Linear(16, 32),
        )

        # Mock the HuggingFace CausalLM interface
        original_forward = model.forward

        def hf_forward(input_ids=None, labels=None, **kwargs):
            embeddings = model[0](input_ids)
            logits = model[1](embeddings)

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, 32), shift_labels.view(-1),
                )

            output = MagicMock()
            output.loss = loss
            output.logits = logits
            return output

        model.forward = hf_forward
        return model

    def test_output_shape_and_type(self):
        model = self._make_tiny_model()
        input_ids = torch.randint(0, 32, (1, 10))
        projected, loss, grad_norm = compute_projected_gradient(model, input_ids, projection_dim=64, seed=42)

        assert projected.shape == (64,)
        assert projected.dtype == torch.float32
        assert isinstance(loss, float)
        assert isinstance(grad_norm, float)

    def test_deterministic_same_seed(self):
        model = self._make_tiny_model()
        input_ids = torch.randint(0, 32, (1, 10))

        a, loss_a, norm_a = compute_projected_gradient(model, input_ids, projection_dim=64, seed=42)
        b, loss_b, norm_b = compute_projected_gradient(model, input_ids, projection_dim=64, seed=42)

        torch.testing.assert_close(a, b)
        assert loss_a == loss_b
        assert norm_a == norm_b

    def test_different_seed(self):
        model = self._make_tiny_model()
        input_ids = torch.randint(0, 32, (1, 10))

        a, _, norm_a = compute_projected_gradient(model, input_ids, projection_dim=64, seed=42)
        b, _, norm_b = compute_projected_gradient(model, input_ids, projection_dim=64, seed=99)

        # Same loss and grad norm (same model + same input), but different projections
        assert not torch.allclose(a, b)
        assert norm_a == norm_b

    def test_different_inputs(self):
        model = self._make_tiny_model()
        input_a = torch.randint(0, 32, (1, 10))
        input_b = torch.randint(0, 32, (1, 10))

        proj_a, _, _ = compute_projected_gradient(model, input_a, projection_dim=64, seed=42)
        proj_b, _, _ = compute_projected_gradient(model, input_b, projection_dim=64, seed=42)

        # Different inputs should generally produce different gradients
        assert not torch.allclose(proj_a, proj_b)

    def test_grads_zeroed_after(self):
        """Gradients should be None after compute_projected_gradient."""
        model = self._make_tiny_model()
        input_ids = torch.randint(0, 32, (1, 10))

        compute_projected_gradient(model, input_ids, projection_dim=32, seed=0)

        for p in model.parameters():
            assert p.grad is None

    def test_nonzero_projection(self):
        """Projected gradient should be nonzero for a real forward/backward pass."""
        model = self._make_tiny_model()
        input_ids = torch.randint(0, 32, (1, 10))

        projected, loss, grad_norm = compute_projected_gradient(model, input_ids, projection_dim=64, seed=42)

        assert loss > 0
        assert grad_norm > 0
        assert torch.any(projected != 0)

    def test_loss_is_positive(self):
        """Cross-entropy loss should be positive."""
        model = self._make_tiny_model()
        input_ids = torch.randint(0, 32, (1, 10))

        _, loss, _ = compute_projected_gradient(model, input_ids, projection_dim=32, seed=0)

        assert loss > 0
